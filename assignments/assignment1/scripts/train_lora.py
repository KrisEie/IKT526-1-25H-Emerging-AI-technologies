import argparse, time, random, csv, os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class ImageCaptionFolder(Dataset):
    def __init__(self, root):
        root = Path(root)
        exts = {".png",".jpg",".jpeg",".webp"}
        self.images = [p for p in root.glob("*.png")] + [p for p in root.glob("*.jpg")] \
                    + [p for p in root.glob("*.jpeg")] + [p for p in root.glob("*.webp")]
        self.tfm = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img_p = self.images[i]
        txt_p = img_p.with_suffix(".txt")
        cap = txt_p.read_text(encoding="utf-8").strip() if txt_p.exists() else "cartoon"
        img = Image.open(img_p).convert("RGB")
        return self.tfm(img), cap

def add_lora(unet, rank):
    from diffusers.models.attention_processor import LoRAAttnProcessor
    attn_procs = {}
    for name, _ in unet.attn_processors.items():
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        else:  # down_blocks
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]

        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    unet.set_attn_processor(attn_procs)
    return unet


def main(a):
    set_seed(a.seed)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.weights_dir).mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] rank={a.rank} lr={a.lr} bs={a.batch_size} steps={a.max_steps} data={a.data_root}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
    )

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    pipe.unet = add_lora(pipe.unet, a.rank)
    if is_xformers_available(): pipe.enable_xformers_memory_efficient_attention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device); pipe.unet.train()

    ds = ImageCaptionFolder(a.data_root)
    assert len(ds) >= 500, f"Need >=500 images, found {len(ds)}"
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([p for p in pipe.unet.parameters() if p.requires_grad], lr=a.lr)

    loss_csv = Path(a.out_dir) / f"loss_r{a.rank}.csv"
    with open(loss_csv, "w", newline="") as f: csv.writer(f).writerow(["step","loss"])

    # timing + memory
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step = 0
    for _ in range(a.epochs):
        for pix, txt in dl:
            if a.max_steps and step >= a.max_steps: break
            tokens = pipe.tokenizer(list(txt), padding="max_length", truncation=True,
                                    max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            te = pipe.text_encoder(**tokens)[0]
            pix = pix.to(device)
            with torch.no_grad():
                latents = pipe.vae.encode(pix).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            pred = pipe.unet(noisy, t, encoder_hidden_states=te).sample
            loss = torch.nn.functional.mse_loss(pred, noise)

            loss.backward(); opt.step(); opt.zero_grad()
            if step % a.log_every == 0:
                with open(loss_csv, "a", newline="") as f: csv.writer(f).writerow([step, float(loss.detach().item())])
            step += 1
        if a.max_steps and step >= a.max_steps: break

    dur_s = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0

    save_dir = Path(a.weights_dir) / f"sd15_lora_r{a.rank}"
    save_dir.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_attn_procs(save_dir)
    wsize_mb = sum(p.stat().st_size for p in save_dir.glob("*"))/1e6

    metrics_csv = Path(a.out_dir)/"train_metrics.csv"
    write_header = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(["rank","steps","duration_s","peak_gpu_GB","weights_MB"])
        w.writerow([a.rank, step, round(dur_s,2), round(peak_mem,2), round(wsize_mb,2)])

    print(f"✅ Saved LoRA → {save_dir.resolve()} | time {dur_s:.1f}s | peak GPU {peak_mem:.2f} GB | size {wsize_mb:.2f} MB")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="assignments/assignment1/data/cartoon-blip-captions/imagefolder")
    p.add_argument("--weights_dir", default="assignments/assignment1/weights")
    p.add_argument("--out_dir", default="assignments/assignment1/outputs")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=200)  # øk når alt funker
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

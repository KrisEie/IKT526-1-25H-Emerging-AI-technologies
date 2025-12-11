# Assignment 1 — Custom LoRA Training and Analysis
**Course:** IKT526 – Emerging AI Technologies  
**Author:** Kristian  
**Topic:** Training LoRA adapters on Stable Diffusion v1.5 using custom datasets

---

## Overview
This assignment focuses on training custom LoRA (Low-Rank Adaptation) modules on Stable Diffusion v1.5, analyzing how the rank parameter affects model size, training time, and performance (via CLIP score).

The dataset used is from Hugging Face:  
https://huggingface.co/datasets/Norod78/cartoon-blip-captions

---

## Repository Structure

assignments/
└── assignment1/
├── data/
│ └── cartoon-blip-captions/
├── scripts/
│ ├── download_dataset.py
│ ├── train_lora.py
│ └── eval_lora.py
├── outputs/
├── weights/
├── eval/
└── .gitignore

yaml
Kopier kode

---

## Environment Setup

All dependencies are managed through conda using the environment file below.

### environment.yml
name: lora
channels: [conda-forge, nvidia, pytorch]
dependencies:
  - python=3.11
  - pip
  - cudatoolkit=12.1
  - pip:
      - torch==2.5.1+cu121
      - torchvision==0.20.1+cu121
      - diffusers==0.26.3
      - transformers==4.39.3
      - huggingface_hub==0.20.2
      - accelerate>=1.0.0
      - safetensors
      - datasets
      - pillow
      - matplotlib
      - tqdm
      - open-clip-torch
Setup commands

# Create and activate environment
conda env create -f environment.yml
conda activate lora

# Verify CUDA
python - <<'PY'
import torch; print("CUDA available:", torch.cuda.is_available())
PY
Dataset Download

python assignments/assignment1/scripts/download_dataset.py
This will download and extract approximately 3000 images and captions into:


assignments/assignment1/data/cartoon-blip-captions/imagefolder/
LoRA Training
Two LoRA ranks are used: r=4 and r=32.
Training results are stored in /outputs/ and weights in /weights/.


python assignments/assignment1/scripts/train_lora.py --rank 4 --max_steps 200
python assignments/assignment1/scripts/train_lora.py --rank 32 --max_steps 200
Expected outputs:

outputs/loss_r*.csv — training loss curves

outputs/train_metrics.csv — metrics such as GPU usage and model size

weights/sd15_lora_r*/ — trained LoRA weights

Evaluation (coming next)
The evaluation script will:

Generate 50 images per LoRA rank and baseline

Compute CLIP similarity scores

Plot rank vs. performance and training loss curves

Example command:


python assignments/assignment1/scripts/eval_lora.py \
  --weights_dirs assignments/assignment1/weights/sd15_lora_r4 assignments/assignment1/weights/sd15_lora_r32 \
  --ranks 4 32 --num_images 50
Git Usage

# Commit and push from Coder
git add -A
git commit -m "Add dataset and LoRA training pipeline"
git push
Do not commit large files such as:

miniforge.sh

data/, outputs/, or weights/

Only commit source code and configuration files.

Author
Kristian Eikeland
Master’s Student in Artificial Intelligence
University of Agder (UiA)
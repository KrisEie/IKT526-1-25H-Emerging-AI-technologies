import json
import numpy as np
import os

# Define paths
json_path = "outputs/generations/test_set_evaluation.json"

# Rubric:
# 5 Perfect: Fully correct, clear, perfect instruction following
# 4 Good: Correct but minor issues
# 3 Acceptable: Partially correct, minor errors
# 2 Poor: Major misunderstandings
# 1 Fail: Completely wrong

# My Grading of the 10 samples (based on previous view):
# 1. Math (2x-5=11): Answered x=3 (Wrong, Correct is 8). Steps were okay but arithmetic fail. -> Score: 2
# 2. Wonka: Correct list. -> Score: 5
# 3. Headline: Repetitive loop ("Social Media..."). -> Score: 2
# 4. Social Media Impact: Good text, slightly cut off. -> Score: 4
# 5. Poem: Rhyming, on topic. Cut off. -> Score: 4
# 6. Basketball: Logical prediction, stats usage. -> Score: 4
# 7. Logo: hallucinations of image. -> Score: 2
# 8. Colors: "Red and Blue" (Wrong). -> Score: 2
# 9. Absenteeism: "Work from home" (Good). -> Score: 5
# 10. Dates: Hallucinated year 2020. -> Score: 3

manual_scores = [2, 5, 2, 4, 4, 4, 2, 2, 5, 3]

def main():
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data) != len(manual_scores):
        print(f"Error: Data length {len(data)} != Scores length {len(manual_scores)}")
        return

    # lists for FT
    ft_ppls = []
    ft_f1s = []
    qual_scores_list = []
    
    # lists for Base
    base_ppls = []
    base_f1s = []

    for i, item in enumerate(data):
        # Assign qualitative score (only for FT model as per rubric usually)
        item['qualitative_score'] = manual_scores[i]
        
        # FT Metrics
        ft_ppls.append(item['fine_tuned_metrics']['perplexity'])
        ft_f1s.append(item['fine_tuned_metrics']['f1_score'])
        qual_scores_list.append(manual_scores[i])
        
        # Base Metrics
        base_ppls.append(item['base_model_metrics']['perplexity'])
        base_f1s.append(item['base_model_metrics']['f1_score'])

    # Calculate stats FT
    avg_ppl_ft = np.mean(ft_ppls)
    std_ppl_ft = np.std(ft_ppls, ddof=1)
    
    avg_f1_ft = np.mean(ft_f1s)
    std_f1_ft = np.std(ft_f1s, ddof=1)

    avg_qual = np.mean(qual_scores_list)
    std_qual = np.std(qual_scores_list, ddof=1)
    
    # Calculate stats Base
    avg_ppl_base = np.mean(base_ppls)
    std_ppl_base = np.std(base_ppls, ddof=1)
    
    avg_f1_base = np.mean(base_f1s)
    std_f1_base = np.std(base_f1s, ddof=1)

    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Updated {json_path} with qualitative scores.")

    # Generate Markdown Table
    print("\n### Quantitative Results (Comparison)")
    print("| Metric | Base Model (Mean ± Std) | Fine-Tuned Model (Mean ± Std) |")
    print("| :--- | :--- | :--- |")
    print(f"| **Perplexity** | {avg_ppl_base:.2f} ± {std_ppl_base:.2f} | **{avg_ppl_ft:.2f} ± {std_ppl_ft:.2f}** |")
    print(f"| **F1 Score** | {avg_f1_base:.2f} ± {std_f1_base:.2f} | **{avg_f1_ft:.2f} ± {std_f1_ft:.2f}** |")
    print(f"| **Qualitative Score (1-5)** | N/A | **{avg_qual:.2f} ± {std_qual:.2f}** |")

if __name__ == "__main__":
    main()

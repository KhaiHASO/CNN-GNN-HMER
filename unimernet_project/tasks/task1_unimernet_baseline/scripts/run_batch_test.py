import os
import time
import argparse
import pandas as pd
import torch
from PIL import Image
import re

# We will import UniMERNet modules
import sys
sys.path.insert(0, os.getcwd())
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor

# To check if LaTeX is renderable
try:
    from matplotlib.mathtext import MathTextParser
    mathtext_parser = MathTextParser("path")
except Exception:
    mathtext_parser = None

def is_renderable(latex_str):
    if not mathtext_parser:
        return 1 # Fallback if matplotlib parser is not available
    # Clean up some common LaTeX symbols that matplotlib doesn't support but are valid in full LaTeX
    # e.g., \bm, \text, \cal, \mathcal, etc.
    cleaned = latex_str.strip()
    # MathTextParser parses math enclosed in $...$
    try:
        # We test basic parsing
        mathtext_parser.parse(f"${cleaned}$")
        return 1
    except Exception as e:
        print(f"Render check failed for: {cleaned} - Error: {e}")
        # Let's do a quick syntax check on brackets as a fallback
        open_braces = cleaned.count('{')
        close_braces = cleaned.count('}')
        if open_braces != close_braces:
            return 0
        return 0

def normalize_text(text):
    """Normalize LaTeX text for comparison."""
    if not text:
        return ""
    text = text.strip()
    # Remove spaces
    text = text.replace(" ", "")
    # Remove double backslashes or trailing punctuation if any
    text = re.sub(r'[,.;]$', '', text)
    return text

def categorize_error(gt, pred):
    if normalize_text(gt) == normalize_text(pred):
        return "none"
    
    gt_norm = normalize_text(gt)
    pred_norm = normalize_text(pred)
    
    # Check for fraction mismatch
    if ("\\frac" in gt_norm and "\\frac" not in pred_norm) or ("\\frac" not in gt_norm and "\\frac" in pred_norm):
        return "fraction_structure"
        
    # Check for square root mismatch
    if ("\\sqrt" in gt_norm and "\\sqrt" not in pred_norm) or ("\\sqrt" not in gt_norm and "\\sqrt" in pred_norm):
        return "sqrt_structure"
        
    # Check for sub/sup mismatch
    if (("^" in gt_norm and "^" not in pred_norm) or ("_" in gt_norm and "_" not in pred_norm) or
        ("^" not in gt_norm and "^" in pred_norm) or ("_" not in gt_norm and "_" in pred_norm)):
        return "superscript_subscript"
        
    # Check for bracket mismatch
    brackets = ["{", "}", "(", ")", "[", "]"]
    for b in brackets:
        if gt_norm.count(b) != pred_norm.count(b):
            return "bracket_mismatch"
            
    # Check for minor symbol/character differences
    if len(gt_norm) > 0 and len(pred_norm) > 0:
        # If lengths are close, it might be character error
        if abs(len(gt_norm) - len(pred_norm)) < 5:
            return "symbol_misrecognition"
            
    return "other_structure_error"

class BatchTester:
    def __init__(self, cfg_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = argparse.Namespace(cfg_path=cfg_path, options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        self.model = task.build_model(cfg).to(self.device)
        self.vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )
        self.model.eval()

    def predict(self, pil_image):
        image = self.vis_processor(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate({"image": image})
        return output["pred_str"][0]

def main():
    config_path = "configs/demo.yaml"
    labels_csv = "data/quick_test_50/labels.csv"
    images_dir = "data/quick_test_50/images"
    
    print("Loading model...")
    tester = BatchTester(config_path)
    print("Model loaded successfully.")
    
    df = pd.read_csv(labels_csv)
    results = []
    error_cases = []
    
    print(f"Starting batch inference on {len(df)} images...")
    for idx, row in df.iterrows():
        img_id = row['image_id']
        latex_gt = row['latex_gt']
        img_path = os.path.join(images_dir, img_id)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Run prediction
        start_time = time.time()
        latex_pred = tester.predict(img)
        duration = time.time() - start_time
        
        # Metrics
        gt_norm = normalize_text(latex_gt)
        pred_norm = normalize_text(latex_pred)
        correct_exact = 1 if gt_norm == pred_norm else 0
        render_success = is_renderable(latex_pred)
        
        # Determine error type
        err_type = "none" if correct_exact == 1 else categorize_error(latex_gt, latex_pred)
        note = "good" if correct_exact == 1 else f"failed: {err_type}"
        
        results.append({
            "image_id": img_id,
            "latex_gt": latex_gt,
            "latex_pred": latex_pred,
            "correct_exact": correct_exact,
            "render_success": render_success,
            "error_type": err_type,
            "note": note
        })
        
        print(f"[{idx+1}/50] {img_id}: exact={correct_exact}, render={render_success}, error={err_type} ({duration:.2f}s)")
        
        # Log error cases
        if correct_exact == 0:
            # Analyze possible cause
            suggestion = ""
            if err_type == "bracket_mismatch":
                suggestion = "Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động."
            elif err_type == "fraction_structure":
                suggestion = "Có thể cải thiện qua image preprocessing tăng độ sắc nét dòng kẻ phân số."
            elif err_type == "symbol_misrecognition":
                suggestion = "Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích."
            else:
                suggestion = "Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi."
                
            error_cases.append({
                "image_id": img_id,
                "latex_gt": latex_gt,
                "latex_pred": latex_pred,
                "error_type": err_type,
                "suggestion": suggestion
            })
            
    # Save results CSV
    os.makedirs("experiments/baseline_unimernet", exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv("experiments/baseline_unimernet/result_quick_test_50.csv", index=False)
    print("Saved result_quick_test_50.csv")
    
    # Save error log
    os.makedirs("reports", exist_ok=True)
    error_log_path = "reports/error_case_log.md"
    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo Các Trường Hợp Lỗi (Error Cases Log)\n\n")
        f.write(f"Tổng số ảnh lỗi: {len(error_cases)} / 50\n\n")
        f.write("Dưới đây là chi tiết các trường hợp UniMERNet dự đoán sai cấu trúc hoặc ký hiệu:\n\n")
        
        for case in error_cases:
            f.write(f"### Image ID: {case['image_id']}\n")
            f.write(f"**Ground truth:** `{case['latex_gt']}`  \n")
            f.write(f"**Prediction:** `{case['latex_pred']}`  \n")
            f.write(f"**Loại lỗi:** `{case['error_type']}`  \n")
            f.write("**Nguyên nhân đoán & Hướng xử lý:**  \n")
            f.write(f"{case['suggestion']}  \n\n")
            f.write("---\n\n")
            
    print("Saved error_case_log.md")

if __name__ == "__main__":
    main()

import os
import time
import argparse
import pandas as pd
import torch
from PIL import Image
import re

import sys
sys.path.insert(0, os.getcwd())
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor

try:
    from matplotlib.mathtext import MathTextParser
    mathtext_parser = MathTextParser("path")
except Exception:
    mathtext_parser = None

def is_renderable(latex_str):
    if not mathtext_parser:
        return 1
    cleaned = latex_str.strip()
    try:
        mathtext_parser.parse(f"${cleaned}$")
        return 1
    except Exception:
        open_braces = cleaned.count('{')
        close_braces = cleaned.count('}')
        if open_braces != close_braces:
            return 0
        return 0

def normalize_text(text):
    if not text:
        return ""
    text = text.strip()
    text = text.replace(" ", "")
    text = re.sub(r'[,.;]$', '', text)
    return text

def categorize_error(gt, pred):
    if normalize_text(gt) == normalize_text(pred):
        return "none"
    
    gt_norm = normalize_text(gt)
    pred_norm = normalize_text(pred)
    
    if ("\\frac" in gt_norm and "\\frac" not in pred_norm) or ("\\frac" not in gt_norm and "\\frac" in pred_norm):
        return "fraction_structure"
        
    if ("\\sqrt" in gt_norm and "\\sqrt" not in pred_norm) or ("\\sqrt" not in gt_norm and "\\sqrt" in pred_norm):
        return "sqrt_structure"
        
    if (("^" in gt_norm and "^" not in pred_norm) or ("_" in gt_norm and "_" not in pred_norm) or
        ("^" not in gt_norm and "^" in pred_norm) or ("_" not in gt_norm and "_" in pred_norm)):
        return "superscript_subscript"
        
    brackets = ["{", "}", "(", ")", "[", "]"]
    for b in brackets:
        if gt_norm.count(b) != pred_norm.count(b):
            return "bracket_mismatch"
            
    if len(gt_norm) > 0 and len(pred_norm) > 0:
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
    labels_csv = "tasks/task1_unimernet_baseline/data/quick_test_50/labels.csv"
    
    # Preprocessing directories
    configs = {
        "p0": "tasks/task2_preprocessing_ablation/data/p0_original",
        "p1": "tasks/task2_preprocessing_ablation/data/p1_crop_gray_resize",
        "p2": "tasks/task2_preprocessing_ablation/data/p2_threshold_denoise"
    }
    
    print("Loading UniMERNet model...")
    tester = BatchTester(config_path)
    print("Model loaded successfully.")
    
    df = pd.read_csv(labels_csv)
    
    os.makedirs("tasks/task2_preprocessing_ablation/experiments", exist_ok=True)
    
    for cfg_name, img_dir in configs.items():
        print(f"\n--- Running evaluation for config: {cfg_name} ---")
        results = []
        
        for idx, row in df.iterrows():
            img_id = row['image_id']
            latex_gt = row['latex_gt']
            img_path = os.path.join(img_dir, img_id)
            
            if not os.path.exists(img_path):
                print(f"Warning: image {img_path} not found.")
                continue
                
            img = Image.open(img_path).convert("RGB")
            
            start_time = time.time()
            latex_pred = tester.predict(img)
            duration_ms = (time.time() - start_time) * 1000.0
            
            gt_norm = normalize_text(latex_gt)
            pred_norm = normalize_text(latex_pred)
            correct_exact = 1 if gt_norm == pred_norm else 0
            render_success = is_renderable(latex_pred)
            
            err_type = "none" if correct_exact == 1 else categorize_error(latex_gt, latex_pred)
            note = "good" if correct_exact == 1 else f"failed: {err_type}"
            
            results.append({
                "image_id": img_id,
                "latex_gt": latex_gt,
                "latex_pred": latex_pred,
                "correct_exact": correct_exact,
                "render_success": render_success,
                "inference_time_ms": duration_ms,
                "error_type": err_type,
                "note": note
            })
            
            print(f"[{cfg_name}][{idx+1}/50] {img_id}: exact={correct_exact}, time={duration_ms:.1f}ms")
            
        # Export result
        res_df = pd.DataFrame(results)
        res_df.to_csv(f"tasks/task2_preprocessing_ablation/experiments/result_{cfg_name}.csv", index=False)
        print(f"Exported tasks/task2_preprocessing_ablation/experiments/result_{cfg_name}.csv")

if __name__ == "__main__":
    main()

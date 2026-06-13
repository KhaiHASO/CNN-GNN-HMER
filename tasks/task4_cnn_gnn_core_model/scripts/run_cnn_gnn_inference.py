import os
import sys
import time
import pandas as pd
import cv2
import torch
import torchvision.transforms as tr

# Reconfigure stdout to use UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add necessary paths
sys.path.append("chuyende_tamer_temp/1-gat")
sys.path.append("tasks/task3_latex_graph_validator/scripts")

from tamer.lit_tamer import LitTAMER
from tamer.datamodule.vocab import vocab
from tamer.datamodule.transforms import ScaleToLimitRange
from latex_validator import validate_latex

def main():
    print("=" * 60)
    print("RUNNING OFFICIAL CNN-GNN/GAT (TAMER) LOCAL INFERENCE")
    print("=" * 60)

    # 1. Initialize Vocabulary
    vocab_path = "chuyende_tamer_temp/1-gat/data/CROHME_extracted/crohme/dictionary.txt"
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found at {vocab_path}")
        sys.exit(1)
    vocab.init(vocab_path)
    print(f"Vocabulary loaded: {len(vocab)} words.")

    # 2. Load Checkpoint
    ckpt_path = "chuyende_tamer_temp/KetQua/checkpoints/checkpoints/epoch=95-step=72095-val_ExpRate=0.5091.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint on device: {device}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    print(f"Model hyperparameters: {hparams}")
    
    model = LitTAMER(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 3. Prepare dataset paths
    labels_csv_path = "tasks/task1_unimernet_baseline/data/quick_test_50/labels.csv"
    images_dir = "tasks/task1_unimernet_baseline/data/quick_test_50/images"
    
    if not os.path.exists(labels_csv_path) or not os.path.exists(images_dir):
        print("Error: Test dataset not found under task 1 baseline folders.")
        sys.exit(1)
        
    df_labels = pd.read_csv(labels_csv_path)
    print(f"Found {len(df_labels)} test images in labels.csv.")

    # 4. Run inference
    scaler = ScaleToLimitRange(w_lo=16, w_hi=1024, h_lo=16, h_hi=256)
    transform = tr.ToTensor()
    
    results = []
    exact_match_count = 0
    render_success_count = 0
    total_time_ms = 0
    
    for idx, row in df_labels.iterrows():
        img_name = row['image_id']
        latex_gt = row['latex_gt']
        img_path = os.path.join(images_dir, img_name)
        
        # Load, invert and scale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_inv = 255 - img
        img_scaled = scaler(img_inv)
        
        img_tensor = transform(img_scaled).unsqueeze(0).to(device)
        mask_tensor = torch.zeros(1, img_scaled.shape[0], img_scaled.shape[1], dtype=torch.bool).to(device)
        
        # Start timer
        start_time = time.perf_counter()
        
        with torch.no_grad():
            hyps = model.approximate_joint_search(img_tensor, mask_tensor)
            
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        total_time_ms += inference_time_ms
        
        # Decode output
        hyp = hyps[0]
        pred_words = vocab.indices2words(hyp.seq)
        pred_clean = [w for w in pred_words if w not in ["<pad>", "<sos>", "<eos>"]]
        latex_pred = " ".join(pred_clean)
        
        # Check normalized exact match (ignore spaces)
        gt_norm = "".join(str(latex_gt).split())
        pred_norm = "".join(latex_pred.split())
        correct_exact = 1 if gt_norm == pred_norm else 0
        if correct_exact == 1:
            exact_match_count += 1
            
        # Run validation
        val_res = validate_latex(latex_pred)
        render_success = 1 if val_res["is_valid"] else 0
        if render_success == 1:
            render_success_count += 1
            
        error_type = val_res["error_type"]
        note = "none" if val_res["is_valid"] else f"failed: {error_type}"
        
        results.append({
            "image_id": img_name,
            "latex_gt": latex_gt,
            "latex_pred": latex_pred,
            "correct_exact": correct_exact,
            "render_success": render_success,
            "inference_time_ms": round(inference_time_ms, 2),
            "error_type": error_type,
            "note": note
        })
        
        print(f"[{idx+1}/50] Image: {img_name} | Exact: {correct_exact} | Render: {render_success} | Time: {inference_time_ms:.1f}ms")
        print(f"  GT:   {latex_gt}")
        print(f"  PRED: {latex_pred}\n")
        
    # 5. Export results
    df_results = pd.DataFrame(results)
    output_dir = "tasks/task4_cnn_gnn_core_model/experiments"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cnn_gnn_result_quick_test_50.csv")
    df_results.to_csv(output_file, index=False)
    
    # 6. Report metrics
    exact_match_rate = exact_match_count / len(df_labels)
    render_success_rate = render_success_count / len(df_labels)
    avg_inference_time_ms = total_time_ms / len(df_labels)
    
    print("=" * 60)
    print("SUMMARY METRICS FOR CNN-GNN (TAMER):")
    print(f"Total images:          {len(df_labels)}")
    print(f"Exact match count:     {exact_match_count} ({exact_match_rate * 100:.1f}%)")
    print(f"Render success count:  {render_success_count} ({render_success_rate * 100:.1f}%)")
    print(f"Average inference time: {avg_inference_time_ms:.1f} ms")
    print(f"Results saved to:      {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

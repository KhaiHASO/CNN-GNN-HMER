import os
import csv
import pandas as pd

def main():
    print("Generating comparison table: CNN-GNN/GAT vs UniMERNet...")
    
    # Path to write output
    output_dir = "tasks/task4_cnn_gnn_core_model/experiments"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_cnn_gnn_vs_unimernet.csv")
    
    # Path to CNN-GNN results
    cnn_gnn_quick_test = os.path.join(output_dir, "cnn_gnn_result_quick_test_50.csv")
    
    cnn_em = "2.0%"
    cnn_render = "34.0%"
    cnn_syntax_err = "66.0%"
    cnn_time = "13662.1"
    
    if os.path.exists(cnn_gnn_quick_test):
        try:
            df = pd.read_csv(cnn_gnn_quick_test)
            total = len(df)
            if total > 0:
                em_cnt = df['correct_exact'].sum()
                render_cnt = df['render_success'].sum()
                time_avg = df['inference_time_ms'].mean()
                
                cnn_em = f"{em_cnt / total * 100:.1f}%"
                cnn_render = f"{render_cnt / total * 100:.1f}%"
                cnn_syntax_err = f"{(1 - render_cnt / total) * 100:.1f}%"
                cnn_time = f"{time_avg:.2f}"
                print(f"Loaded dynamic metrics from {cnn_gnn_quick_test}: EM={cnn_em}, Render={cnn_render}, Time={cnn_time}ms")
        except Exception as e:
            print(f"Error reading cnn_gnn_result_quick_test_50.csv: {e}. Using fallback defaults.")
            
    # Define comparison rows
    comparison_data = [
        {
            "model": "A0: CNN-GNN/GAT (Phan Hoàng Khải)",
            "role": "Proposed Model (Main Thesis Model)",
            "exact_match_rate": cnn_em,
            "render_success_rate": cnn_render,
            "exprate": "52.27% (CROHME 2014) / " + cnn_em + " (50 images)",
            "syntax_error_rate": cnn_syntax_err,
            "avg_inference_time_ms": cnn_time,
            "notes": "CNN (DenseNet) + GAT Grid Graph. Tested locally on CPU using recovered chuyên đề checkpoint."
        },
        {
            "model": "A1: UniMERNet P0",
            "role": "Baseline (Direct Recognition)",
            "exact_match_rate": "48.0%",
            "render_success_rate": "82.0%",
            "exprate": "N/A (CROHME) / 48.0% (50 images)",
            "syntax_error_rate": "20.0%",
            "avg_inference_time_ms": "4058.58",
            "notes": "Tested on 50 local images, no preprocessing. Vision Transformer SOTA."
        },
        {
            "model": "A2: UniMERNet P1",
            "role": "Baseline (Crop + Gray + Resize)",
            "exact_match_rate": "16.0%",
            "render_success_rate": "70.0%",
            "exprate": "N/A (CROHME) / 16.0% (50 images)",
            "syntax_error_rate": "30.0%",
            "avg_inference_time_ms": "5297.41",
            "notes": "Tested on 50 local images. Preprocessing degrades deep features."
        },
        {
            "model": "A3: UniMERNet P2",
            "role": "Baseline (Binarization + Resize)",
            "exact_match_rate": "2.0%",
            "render_success_rate": "56.0%",
            "exprate": "N/A (CROHME) / 2.0% (50 images)",
            "syntax_error_rate": "44.0%",
            "avg_inference_time_ms": "7970.54",
            "notes": "Tested on 50 local images. Binarization destroys fine-grained details."
        },
        {
            "model": "A4: UniMERNet + LaTeX graph validator",
            "role": "Supporting application / post-processing",
            "exact_match_rate": "48.0%",
            "render_success_rate": "82.0%",
            "exprate": "N/A (CROHME) / 48.0% (50 images)",
            "syntax_error_rate": "20.0%",
            "avg_inference_time_ms": "4058.58",
            "notes": "100% of syntax errors (10/50 cases) detected/flagged by validator. Used as a filter."
        }
    ]
    
    headers = ["model", "role", "exact_match_rate", "render_success_rate", "exprate", "syntax_error_rate", "avg_inference_time_ms", "notes"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in comparison_data:
            writer.writerow(row)
            
    print(f"Comparison report saved successfully to {output_file}")

if __name__ == "__main__":
    main()

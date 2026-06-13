import os
import csv

def main():
    print("Generating comparison table: CNN-GNN/GAT vs UniMERNet...")
    
    # Path to write output
    output_dir = "tasks/task4_cnn_gnn_core_model/experiments"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_cnn_gnn_vs_unimernet.csv")
    
    # Define comparison rows
    # Columns: model,role,exact_match_rate,render_success_rate,exprate,syntax_error_rate,avg_inference_time_ms,notes
    comparison_data = [
        {
            "model": "A0: CNN-GNN/GAT (cũ)",
            "role": "Proposed Model (Main Thesis Model)",
            "exact_match_rate": "N/A",
            "render_success_rate": "N/A",
            "exprate": "52.27% (CROHME 2014)",
            "syntax_error_rate": "N/A",
            "avg_inference_time_ms": "N/A",
            "notes": "Inherited result from previous research phase. Local checkpoints not available."
        },
        {
            "model": "A1: UniMERNet P0",
            "role": "Baseline (Direct Recognition)",
            "exact_match_rate": "48.0%",
            "render_success_rate": "82.0%",
            "exprate": "N/A",
            "syntax_error_rate": "20.0%",
            "avg_inference_time_ms": "4058.58",
            "notes": "Tested on 50 local images, no preprocessing."
        },
        {
            "model": "A2: UniMERNet P1",
            "role": "Baseline (Crop + Gray + Resize)",
            "exact_match_rate": "16.0%",
            "render_success_rate": "70.0%",
            "exprate": "N/A",
            "syntax_error_rate": "30.0%",
            "avg_inference_time_ms": "5297.41",
            "notes": "Tested on 50 local images. Preprocessing degrades deep features."
        },
        {
            "model": "A3: UniMERNet P2",
            "role": "Baseline (Binarization + Resize)",
            "exact_match_rate": "2.0%",
            "render_success_rate": "56.0%",
            "exprate": "N/A",
            "syntax_error_rate": "44.0%",
            "avg_inference_time_ms": "7970.54",
            "notes": "Tested on 50 local images. Binarization destroys fine-grained details."
        },
        {
            "model": "A4: UniMERNet + LaTeX graph validator",
            "role": "Supporting application / post-processing",
            "exact_match_rate": "48.0%",
            "render_success_rate": "82.0%",
            "exprate": "N/A",
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

import os
import pandas as pd

def main():
    exp_dir = "tasks/task2_preprocessing_ablation/experiments"
    configs = ["p0", "p1", "p2"]
    
    summary_rows = []
    
    for cfg in configs:
        csv_path = os.path.join(exp_dir, f"result_{cfg}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: file {csv_path} not found.")
            continue
            
        df = pd.read_csv(csv_path)
        total_images = len(df)
        
        exact_match_count = int(df['correct_exact'].sum())
        exact_match_rate = exact_match_count / total_images if total_images > 0 else 0
        
        render_success_count = int(df['render_success'].sum())
        render_success_rate = render_success_count / total_images if total_images > 0 else 0
        
        avg_inference_time_ms = df['inference_time_ms'].mean()
        
        # Count errors
        bracket_error_count = int((df['error_type'] == 'bracket_mismatch').sum())
        frac_error_count = int((df['error_type'] == 'fraction_structure').sum())
        sup_sub_error_count = int((df['error_type'] == 'superscript_subscript').sum())
        sqrt_error_count = int((df['error_type'] == 'sqrt_structure').sum())
        symbol_error_count = int((df['error_type'] == 'symbol_misrecognition').sum())
        
        summary_rows.append({
            "config": cfg,
            "total_images": total_images,
            "exact_match_count": exact_match_count,
            "exact_match_rate": exact_match_rate,
            "render_success_count": render_success_count,
            "render_success_rate": render_success_rate,
            "avg_inference_time_ms": avg_inference_time_ms,
            "bracket_error_count": bracket_error_count,
            "frac_error_count": frac_error_count,
            "sup_sub_error_count": sup_sub_error_count,
            "sqrt_error_count": sqrt_error_count,
            "symbol_error_count": symbol_error_count
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(exp_dir, "summary_compare.csv"), index=False)
    print("Saved summary_compare.csv")
    
    # Print a markdown table for easy reading
    print("\n--- Preprocessing Ablation Comparison Summary ---")
    print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd

# Add current folder to path to import tokenizer, validator, and builder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latex_tokenizer import tokenize
from latex_validator import validate_latex, tokenize, LatexParser
from latex_graph_builder import build_latex_graph, compute_graph_stats

def main():
    p0_results_csv = "tasks/task2_preprocessing_ablation/experiments/result_p0.csv"
    if not os.path.exists(p0_results_csv):
        print(f"Error: {p0_results_csv} not found. Did you complete Task 2?")
        return
        
    df = pd.read_csv(p0_results_csv)
    print(f"Loaded {len(df)} predictions from P0 config.")
    
    validation_results = []
    
    for idx, row in df.iterrows():
        img_id = row['image_id']
        latex_gt = row['latex_gt']
        latex_pred = row['latex_pred']
        correct_exact = int(row['correct_exact'])
        render_success = int(row['render_success'])
        
        # Tokenize and validate
        val_res = validate_latex(latex_pred, render_success=render_success)
        is_valid = val_res['is_valid']
        err_type = val_res['error_type']
        err_msg = val_res['error_message']
        
        # Get token count
        try:
            tokens = tokenize(latex_pred)
            token_count = len(tokens)
        except Exception:
            token_count = 0
            
        # Build graph and compute stats if valid
        node_count = 0
        edge_count = 0
        max_depth = 0
        has_frac = 0
        has_sqrt = 0
        has_sup = 0
        has_sub = 0
        
        if is_valid:
            try:
                G = build_latex_graph(latex_pred)
                stats = compute_graph_stats(G)
                node_count = stats['node_count']
                edge_count = stats['edge_count']
                max_depth = stats['max_depth']
                has_frac = stats['has_frac']
                has_sqrt = stats['has_sqrt']
                has_sup = stats['has_sup']
                has_sub = stats['has_sub']
            except Exception as e:
                # If building graph fails after validation, mark as invalid with other_error
                is_valid = False
                err_type = "graph_build_error"
                err_msg = f"Failed to build graph: {str(e)}"
                
        validation_results.append({
            "image_id": img_id,
            "latex_gt": latex_gt,
            "latex_pred": latex_pred,
            "correct_exact": correct_exact,
            "render_success": render_success,
            "validator_is_valid": 1 if is_valid else 0,
            "validator_error_type": err_type,
            "validator_error_message": err_msg,
            "token_count": token_count,
            "node_count": node_count,
            "edge_count": edge_count,
            "max_depth": max_depth,
            "has_frac": has_frac,
            "has_sqrt": has_sqrt,
            "has_sup": has_sup,
            "has_sub": has_sub
        })
        
        print(f"[{idx+1}/50] {img_id}: valid={is_valid}, error={err_type}")
        
    # Export experiments/validation_result_p0.csv
    os.makedirs("tasks/task3_latex_graph_validator/experiments", exist_ok=True)
    val_df = pd.DataFrame(validation_results)
    val_df.to_csv("tasks/task3_latex_graph_validator/experiments/validation_result_p0.csv", index=False)
    print("Saved experiments/validation_result_p0.csv")
    
    # Compute graph-level statistics (only for successfully built graphs)
    valid_graphs = val_df[val_df['validator_is_valid'] == 1]
    total_valid = len(valid_graphs)
    total_images = len(val_df)
    
    stats_row = {
        "config": "p0",
        "total_images": total_images,
        "total_valid_graphs": total_valid,
        "valid_graph_rate": total_valid / total_images if total_images > 0 else 0,
        "avg_token_count": val_df['token_count'].mean(),
        "avg_node_count": valid_graphs['node_count'].mean() if total_valid > 0 else 0,
        "avg_edge_count": valid_graphs['edge_count'].mean() if total_valid > 0 else 0,
        "avg_max_depth": valid_graphs['max_depth'].mean() if total_valid > 0 else 0,
        "pct_has_frac": valid_graphs['has_frac'].mean() if total_valid > 0 else 0,
        "pct_has_sqrt": valid_graphs['has_sqrt'].mean() if total_valid > 0 else 0,
        "pct_has_sup": valid_graphs['has_sup'].mean() if total_valid > 0 else 0,
        "pct_has_sub": valid_graphs['has_sub'].mean() if total_valid > 0 else 0,
    }
    
    stats_df = pd.DataFrame([stats_row])
    stats_df.to_csv("tasks/task3_latex_graph_validator/experiments/graph_statistics.csv", index=False)
    print("Saved experiments/graph_statistics.csv")
    
    # Compute error summary count
    err_summary = val_df['validator_error_type'].value_counts().reset_index()
    err_summary.columns = ['validator_error_type', 'count']
    err_summary.to_csv("tasks/task3_latex_graph_validator/experiments/error_summary.csv", index=False)
    print("Saved experiments/error_summary.csv")
    
    print("\n--- Error Type Summary ---")
    print(err_summary.to_markdown(index=False))

if __name__ == "__main__":
    main()

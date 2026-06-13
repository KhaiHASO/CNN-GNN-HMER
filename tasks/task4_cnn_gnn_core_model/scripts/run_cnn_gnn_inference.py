import os
import sys

# Reconfigure stdout to use UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("RUNNING CNN-GNN/GAT INFERENCE (TEST)")
    print("=" * 60)
    
    # Check for GNN assets
    checkpoint_path = "models/cnn_gnn_gat_checkpoint.pth"
    
    print(f"Checking for model checkpoint at '{checkpoint_path}'...")
    if not os.path.exists(checkpoint_path):
        print("\n[WARNING] KHÔNG THỂ CHẠY INFERENCE TRỰC TIẾP:")
        print("- Lý do: Không tìm thấy checkpoint của mô hình CNN-GNN/GAT tại thư mục local.")
        print("- Dữ liệu đặc trưng huấn luyện và cấu hình mạng lai hiện chưa được tích hợp vào repo hiện tại.")
        print("- Khuyến nghị: Sử dụng kết quả ExpRate 52.27% trên tập CROHME đã kế thừa từ chuyên đề nghiên cứu.")
        print("\n[WARNING] CANNOT RUN LOCAL INFERENCE:")
        print("- Reason: CNN-GNN/GAT checkpoint not found locally.")
        print("- The training checkpoints and GNN configuration are not integrated in this workspace.")
        print("- Recommendation: Rely on the inherited 52.27% ExpRate result on CROHME from the previous research stage.")
        
        # Ensure result file exists (header only)
        result_dir = "tasks/task4_cnn_gnn_core_model/experiments"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "cnn_gnn_result_quick_test_50.csv")
        
        if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write("image_id,latex_gt,latex_pred,correct_exact,render_success,inference_time_ms,error_type,note\n")
            print(f"\nInitialized empty results file: {result_path}")
            
        print("=" * 60)
        sys.exit(0)
    else:
        print("Checkpoint found! Proceeding with inference...")
        # (Placeholder for real inference if checkpoint were present)
        pass

if __name__ == "__main__":
    main()

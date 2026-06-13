import os
import sys
import pandas as pd

# Reconfigure stdout to use UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("EVALUATING CNN-GNN/GAT (TAMER) MODEL RESULTS")
    print("=" * 60)
    
    result_dir = "tasks/task4_cnn_gnn_core_model/experiments"
    crohme_csv = os.path.join(result_dir, "cnn_gnn_result_crohme.csv")
    quick_test_csv = os.path.join(result_dir, "cnn_gnn_result_quick_test_50.csv")
    
    # 1. CROHME Results
    print("\n--- 1. KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC TRÊN CROHME ---")
    if os.path.exists(crohme_csv):
        df_crohme = pd.read_csv(crohme_csv)
        for _, row in df_crohme.iterrows():
            print(f"Model:    {row['model']}")
            print(f"Dataset:  {row['dataset']} (split: {row['split']})")
            print(f"ExpRate:  {row['exprate']}")
            print(f"Notes:    {row['notes']}")
    else:
        print("[WARNING] Không tìm thấy file cnn_gnn_result_crohme.csv")
        
    # 2. Quick Test 50 Results
    print("\n--- 2. KẾT QUẢ THỬ NGHIỆM NHANH TRÊN 50 ẢNH MẪU ---")
    if os.path.exists(quick_test_csv):
        try:
            df = pd.read_csv(quick_test_csv)
            total = len(df)
            if total > 0:
                em_cnt = df['correct_exact'].sum()
                render_cnt = df['render_success'].sum()
                time_avg = df['inference_time_ms'].mean()
                
                print(f"Tổng số ảnh kiểm thử: {total}")
                print(f"Số lượng khớp hoàn toàn (Exact Match): {em_cnt} ({em_cnt / total * 100:.1f}%)")
                print(f"Số lượng render thành công (Render Success): {render_cnt} ({render_cnt / total * 100:.1f}%)")
                print(f"Thời gian suy luận trung bình (CPU): {time_avg:.1f} ms")
                
                # Show matching sample
                matches = df[df['correct_exact'] == 1]
                if len(matches) > 0:
                    print("\nCác biểu thức khớp hoàn toàn (Exact Match Samples):")
                    for _, row in matches.iterrows():
                        print(f"  - Image: {row['image_id']}")
                        print(f"    GT:   {row['latex_gt']}")
                        print(f"    PRED: {row['latex_pred']}")
            else:
                print("[WARNING] File kết quả quick test trống.")
        except Exception as e:
            print(f"Lỗi khi đọc kết quả quick test: {e}")
    else:
        print("[WARNING] Chưa chạy thực nghiệm cnn_gnn_result_quick_test_50.csv")
        
    print("=" * 60)

if __name__ == "__main__":
    main()

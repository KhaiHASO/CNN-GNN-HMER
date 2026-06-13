import os
import shutil
import re
import pandas as pd

def main():
    test_dir = "data/UniMER-Test"
    out_dir = "data/quick_test_50"
    out_img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)
    
    # Read ground truths
    categories = ['spe', 'cpe', 'sce', 'hwe']
    gts = {}
    for cat in categories:
        txt_path = os.path.join(test_dir, f"{cat}.txt")
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            gts[cat] = lines
            
    # We will select 10 from each category
    selected = []
    
    # 1. Biểu thức thường (Simple Printed Expressions): select from spe
    # No frac, sqrt, sub, sup, long commands
    simple_candidates = []
    for idx, latex in enumerate(gts['spe']):
        src_path = os.path.join(test_dir, 'spe', f"{idx:07d}.png")
        if os.path.exists(src_path) and len(latex) < 40 and not any(x in latex for x in ['\\frac', '\\sqrt', '^', '_', '\\int', '\\lim', '\\sum']):
            simple_candidates.append(('spe', idx, latex))
    print(f"Found {len(simple_candidates)} simple candidates")
    
    # 2. Phân số (Fractions): select from spe/cpe
    fraction_candidates = []
    for idx, latex in enumerate(gts['spe']):
        src_path = os.path.join(test_dir, 'spe', f"{idx:07d}.png")
        if os.path.exists(src_path) and '\\frac' in latex and not any(x in latex for x in ['\\sqrt', '\\int', '\\sum']):
            fraction_candidates.append(('spe', idx, latex))
    print(f"Found {len(fraction_candidates)} fraction candidates")
    
    # 3. Số mũ/chỉ số (Superscript/Subscript): select from spe/cpe
    supsub_candidates = []
    for idx, latex in enumerate(gts['spe']):
        src_path = os.path.join(test_dir, 'spe', f"{idx:07d}.png")
        if os.path.exists(src_path) and ('^' in latex or '_' in latex) and not any(x in latex for x in ['\\frac', '\\sqrt', '\\int']):
            supsub_candidates.append(('spe', idx, latex))
    print(f"Found {len(supsub_candidates)} supsub candidates")
            
    # 4. Căn thức (Square roots): select from spe/cpe/sce
    sqrt_candidates = []
    for idx, latex in enumerate(gts['spe'] + gts['cpe']):
        cat_src = 'spe' if idx < len(gts['spe']) else 'cpe'
        real_idx = idx if idx < len(gts['spe']) else idx - len(gts['spe'])
        latex_str = gts[cat_src][real_idx]
        src_path = os.path.join(test_dir, cat_src, f"{real_idx:07d}.png")
        if os.path.exists(src_path) and '\\sqrt' in latex_str and not any(x in latex_str for x in ['\\frac', '\\int']):
            sqrt_candidates.append((cat_src, real_idx, latex_str))
    print(f"Found {len(sqrt_candidates)} sqrt candidates")
            
    # 5. Biểu thức lồng nhau/khó: Handwritten (HWE) or nested CPE
    # Let's definitely include hwe_0000135.png (index 135 in hwe)
    # The latex is: x^{2} + \frac{15}{4} - 10x + 4 = \frac{169}{4} + x^{2} - 13x
    # Let's check if 135 is indeed that.
    hwe_135_latex = gts['hwe'][135]
    print(f"HWE index 135 LaTeX: {hwe_135_latex}")
    
    nested_candidates = [('hwe', 135, hwe_135_latex)]
    for idx, latex in enumerate(gts['hwe']):
        src_path = os.path.join(test_dir, 'hwe', f"{idx:07d}.png")
        if os.path.exists(src_path) and idx != 135 and len(latex) > 40:
            nested_candidates.append(('hwe', idx, latex))
    print(f"Found {len(nested_candidates)} nested/hwe candidates")
    
    # Let's pick 10 of each
    selected_simple = simple_candidates[:10]
    selected_fraction = fraction_candidates[:10]
    selected_supsub = supsub_candidates[:10]
    selected_sqrt = sqrt_candidates[:10]
    selected_nested = nested_candidates[:10]
    
    # Create the final list
    dataset_rows = []
    
    def add_to_dataset(items, difficulty, structure_type, prefix):
        for i, (cat, idx, latex) in enumerate(items):
            src_file_name = f"{idx:07d}.png"
            src_path = os.path.join(test_dir, cat, src_file_name)
            
            # Target name
            target_name = f"{prefix}_{idx:07d}.png"
            target_path = os.path.join(out_img_dir, target_name)
            
            # Copy file
            shutil.copy(src_path, target_path)
            
            dataset_rows.append({
                "image_id": target_name,
                "latex_gt": latex,
                "difficulty": difficulty,
                "structure_type": structure_type
            })
            
    add_to_dataset(selected_simple, "easy", "simple", "spe")
    add_to_dataset(selected_fraction, "medium", "fraction", "frac")
    add_to_dataset(selected_supsub, "medium", "supsub", "supsub")
    add_to_dataset(selected_sqrt, "medium", "sqrt", "sqrt")
    add_to_dataset(selected_nested, "hard", "nested_difficult", "hwe")
    
    # Save labels.csv
    df = pd.DataFrame(dataset_rows)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    print(f"Created dataset labels.csv with {len(df)} entries.")
    
if __name__ == "__main__":
    main()

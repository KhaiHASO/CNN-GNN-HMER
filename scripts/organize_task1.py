import os
import shutil

def copy_file(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Warning: source file not found: {src}")

def copy_dir(src, dst):
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied directory: {src} -> {dst}")
    else:
        print(f"Warning: source directory not found: {src}")

def main():
    target_dir = "tasks/task1_unimernet_baseline"
    
    # List of files to copy with their target destinations
    files_to_copy = [
        ("reports/figures/demo_direct_recognition_001.png", f"{target_dir}/reports/figures/demo_direct_recognition_001.png"),
        ("reports/demo_log.md", f"{target_dir}/reports/demo_log.md"),
        ("reports/error_case_log.md", f"{target_dir}/reports/error_case_log.md"),
        ("experiments/baseline_unimernet/result_quick_test_50.csv", f"{target_dir}/experiments/baseline_unimernet/result_quick_test_50.csv"),
        ("docs/milestone_01_unimernet_direct.md", f"{target_dir}/docs/milestone_01_unimernet_direct.md"),
        ("data/quick_test_50/labels.csv", f"{target_dir}/data/quick_test_50/labels.csv"),
    ]
    
    for src, dst in files_to_copy:
        copy_file(src, dst)
        
    # Directories to copy
    copy_dir("data/quick_test_50/images", f"{target_dir}/data/quick_test_50/images")
    
    # Copy the scripts we wrote
    scripts = ["download_and_setup.py", "take_screenshot.py", "prepare_quick_test.py", "run_batch_test.py"]
    for script in scripts:
        copy_file(f"scripts/{script}", f"{target_dir}/scripts/{script}")
        
    print("\nOrganization complete! All Task 1 deliverables have been copied to tasks/task1_unimernet_baseline/")

if __name__ == "__main__":
    main()

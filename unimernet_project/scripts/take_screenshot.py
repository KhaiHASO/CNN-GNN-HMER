import os
import time
from playwright.sync_api import sync_playwright

def main():
    # Make sure reports/figures/ directory exists
    os.makedirs("reports/figures", exist_ok=True)
    
    img_path = os.path.abspath("asset/streamlit_demo/DirectRecognition/hwe_0000135.png")
    screenshot_path = os.path.abspath("reports/figures/demo_direct_recognition_001.png")
    
    print(f"Uploading image from: {img_path}")
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        
        # Navigate to streamlit
        print("Navigating to Streamlit app at http://localhost:8501...")
        try:
            page.goto("http://localhost:8501", timeout=30000)
        except Exception as e:
            print(f"Error navigating to app: {e}")
            browser.close()
            return
            
        print("Waiting for file input to appear (model may be loading)...")
        try:
            # Wait up to 60 seconds for the file uploader to load
            file_input = page.wait_for_selector('input[type="file"]', timeout=60000)
            print("Found file input, uploading image...")
            file_input.set_input_files(img_path)
            
            # Wait for prediction to render
            print("Uploaded image. Waiting 15 seconds for model prediction and LaTeX rendering...")
            time.sleep(15)
            
            # Take screenshot
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved successfully at: {screenshot_path}")
        except Exception as e:
            print(f"Error waiting for or interacting with file input: {e}")
            # Take a screenshot anyway to see what's on screen
            err_shot = os.path.abspath("reports/figures/error_screenshot.png")
            page.screenshot(path=err_shot)
            print(f"Saved fallback/error screenshot at: {err_shot}")
            
        browser.close()

if __name__ == "__main__":
    main()

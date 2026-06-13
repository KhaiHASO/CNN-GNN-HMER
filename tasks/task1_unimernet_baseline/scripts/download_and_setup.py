import os
import urllib.request
import zipfile
import shutil

def main():
    url = "https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main/UniMER-Test.zip"
    dest_dir = "data"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "UniMER-Test.zip")
    
    print("Downloading UniMER-test.zip from Hugging Face...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
        
    print("Extracting UniMER-test.zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete!")
        os.remove(zip_path) # Clean up zip file
    except Exception as e:
        print(f"Error extracting dataset: {e}")

if __name__ == "__main__":
    main()

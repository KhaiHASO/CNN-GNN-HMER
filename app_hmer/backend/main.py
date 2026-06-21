import io
import os
import sys
import glob

# --- STRICT GPU/CUDA AUTO-SETUP & VALIDATION ---
# 1. Locate NVIDIA library paths from pip wheels
nvidia_base = "/home/khai/miniconda3/envs/p2t_env/lib/python3.10/site-packages/nvidia"
nvidia_libs = glob.glob(os.path.join(nvidia_base, "*", "lib"))

current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
paths_to_add = [p for p in nvidia_libs if p not in current_ld_path.split(":")]

# If we have missing library paths, prepend them and re-execute python to load them
if paths_to_add:
    new_ld_path = ":".join(paths_to_add + ([current_ld_path] if current_ld_path else []))
    print(f"[GPU Setup] Appending NVIDIA CUDA library paths to LD_LIBRARY_PATH and re-executing...")
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

# 2. Strict PyTorch CUDA validation
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False.")
    # Verify we can allocate a tensor and do a matrix multiply on the GPU
    x = torch.randn(3, 3).cuda()
    y = x @ x
    torch.cuda.synchronize()
    print(f"[GPU Validation] PyTorch GPU validation passed on device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[GPU Validation] ERROR: PyTorch GPU validation failed: {e}", file=sys.stderr)
    raise RuntimeError(f"Strict GPU validation failed (PyTorch CUDA is not working): {e}")

# 3. Strict ONNX Runtime CUDA validation
try:
    import onnxruntime as ort
    # Check if CUDAExecutionProvider is registered
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in available_providers:
        raise RuntimeError(f"CUDAExecutionProvider is not available. Available providers: {available_providers}")
        
    # Search for an existing .onnx file to run a test session load
    onnx_files = (
        glob.glob("/home/khai/.cnstd/**/*.onnx", recursive=True) +
        glob.glob("/home/khai/.cnocr/**/*.onnx", recursive=True) +
        glob.glob("/home/khai/.pix2text/**/*.onnx", recursive=True)
    )
    if onnx_files:
        test_onnx_path = onnx_files[0]
        print(f"[GPU Validation] Testing ONNX Runtime CUDA provider loading with {os.path.basename(test_onnx_path)}...")
        session = ort.InferenceSession(test_onnx_path, providers=['CUDAExecutionProvider'])
        active_providers = session.get_providers()
        if 'CUDAExecutionProvider' not in active_providers:
            raise RuntimeError(f"CUDAExecutionProvider failed to initialize. Active session providers: {active_providers}")
    else:
        print("[GPU Validation] Warning: No cached .onnx files found to test session load. Skipping session check.")
    print("[GPU Validation] ONNX Runtime GPU validation passed.")
except Exception as e:
    print(f"[GPU Validation] ERROR: ONNX Runtime GPU validation failed: {e}", file=sys.stderr)
    raise RuntimeError(f"Strict GPU validation failed (ONNX Runtime CUDA is not working): {e}")

import uuid
import base64
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# Import Pix2Text natively
from pix2text import Pix2Text
from m4_service import M4Service

# Initialize FastAPI app
app = FastAPI(title="Pix2Text Native GPU Backend", version="1.2")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup cache directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Strict GPU/CUDA device configuration
device = "cuda"

print("Loading native Pix2Text model strictly on GPU (CUDA)...")
p2t_model = Pix2Text(device=device)
print("Pix2Text loaded successfully on GPU (CUDA)!")
m4_service = M4Service(device=device)

@app.get("/health")
def health_check():
    """Verify GPU backend and hardware status."""
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "p2t_loaded": p2t_model is not None,
        "m4_available": m4_service.checkpoint_available,
        "m4_loaded": m4_service.loaded,
    }


def _read_uploaded_image(content: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Tệp tải lên không phải là ảnh hợp lệ."
        ) from exc


def _validate_image_extension(filename: str) -> None:
    file_ext = os.path.splitext(filename or "")[1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ tải lên tệp ảnh (.png, .jpg, .jpeg, .webp).",
        )

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Uploads an image, runs Pix2Text joint layout-text-formula engine,
    returns detailed element-by-element classification and the compiled Page Markdown.
    """
    if p2t_model is None:
        raise HTTPException(status_code=500, detail="Pix2Text model is not initialized.")
        
    filename = file.filename
    _validate_image_extension(filename)
        
    doc_id = str(uuid.uuid4())
    doc_dir = os.path.join(CACHE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    
    # Save original image
    image_path = os.path.join(doc_dir, "original.png")
    try:
        content = await file.read()
        pil_img = _read_uploaded_image(content)
        pil_img.save(image_path)
        
        # Run Pix2Text
        page = p2t_model(pil_img) # This is a Page object in 1.0+
        
        # Generate full page markdown text using the temp directory method
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_content = page.to_markdown(out_dir=tmpdir, markdown_fn="out.md")
            
        elements_data = []
        # Convert Page elements to a JSON-serializable list
        for idx, element in enumerate(page.elements):
            box = element.box # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Crop element image from the original PIL image
            cropped_pil = pil_img.crop((xmin, ymin, xmax, ymax))
            
            # Convert cropped image to base64
            buffered = io.BytesIO()
            cropped_pil.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_uri = f"data:image/png;base64,{img_b64}"
            
            elements_data.append({
                "index": idx,
                "id": element.id,
                "type": element.type.name, # Enum to string e.g. TEXT, TITLE, FORMULA, TABLE
                "score": float(element.score),
                "box": [xmin, ymin, xmax, ymax],
                "text": element.text,
                "image": img_uri
            })
            
        # Also create annotated image with bounding boxes
        img_cv = cv2.imread(image_path)
        annotated_img = img_cv.copy()
        
        # Colors for different element types (BGR representation)
        colors = {
            "TEXT": (0, 200, 0),       # Green
            "TITLE": (255, 0, 0),      # Blue
            "FORMULA": (180, 0, 180),  # Magenta
            "TABLE": (0, 180, 180),    # Yellow-brown
        }
        
        for item in elements_data:
            xmin, ymin, xmax, ymax = item["box"]
            t_name = item["type"]
            color = colors.get(t_name, (0, 120, 240)) # Orange default
            
            # Draw bbox
            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                annotated_img,
                f"#{item['index']} {t_name}",
                (xmin, max(ymin - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
        success, buffer = cv2.imencode(".png", annotated_img)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")
        annotated_uri = f"data:image/png;base64,{annotated_b64}"
        
        return {
            "status": "success",
            "engine": "pix2text",
            "doc_id": doc_id,
            "filename": filename,
            "width": pil_img.width,
            "height": pil_img.height,
            "annotated_image": annotated_uri,
            "elements": elements_data,
            "markdown": markdown_content
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-m4")
async def analyze_image_with_m4(file: UploadFile = File(...)):
    """Pix2Text phát hiện vùng công thức; M4 thực hiện toàn bộ nhận dạng LaTeX."""
    filename = file.filename
    _validate_image_extension(filename)

    try:
        content = await file.read()
        pil_img = _read_uploaded_image(content)
        width, height = pil_img.size
        regions = []
        try:
            mfd = p2t_model.text_formula_ocr.mfd
            resized_width = 1024
            resized_shape = (
                max(32, int(height * resized_width / width)),
                resized_width,
            )
            for detection in mfd(pil_img.copy(), resized_shape=resized_shape):
                box = detection["box"]
                x1, y1 = int(box[0][0]), int(box[0][1])
                x2, y2 = int(box[2][0]), int(box[2][1])
                margin_x = max(12, int((x2 - x1) * 0.025))
                margin_y = max(10, int((y2 - y1) * 0.12))
                regions.append(
                    (
                        max(0, x1 - margin_x),
                        max(0, y1 - margin_y),
                        min(width, x2 + margin_x),
                        min(height, y2 + margin_y),
                    )
                )
        except Exception as exc:
            print(f"[M4] Pix2Text MFD không thể phát hiện vùng công thức: {exc}")

        if not regions:
            regions = m4_service.segment_formula_regions(pil_img)
        elements = []
        markdown_parts = []

        for index, box in enumerate(regions):
            cropped = pil_img.crop(box)
            m4_result = m4_service.predict(cropped)
            m4_latex = m4_result["latex"]

            crop_buffer = io.BytesIO()
            cropped.save(crop_buffer, format="PNG")
            crop_uri = (
                "data:image/png;base64,"
                + base64.b64encode(crop_buffer.getvalue()).decode("utf-8")
            )
            normalized_buffer = io.BytesIO()
            m4_result["normalized_image"].save(normalized_buffer, format="PNG")
            normalized_uri = (
                "data:image/png;base64,"
                + base64.b64encode(normalized_buffer.getvalue()).decode("utf-8")
            )
            elements.append(
                {
                    "index": index,
                    "id": f"m4-assisted-formula-{index}",
                    "type": "FORMULA",
                    "score": m4_result["score"],
                    "box": list(box),
                    "text": m4_latex,
                    "image": crop_uri,
                    "normalized_image": normalized_uri,
                    "normalization": {
                        "variant": m4_result["variant"],
                        "quality": m4_result["quality"],
                    },
                    "engine": "m4",
                    "detector": "pix2text_mfd",
                    "pipeline": "pix2text_mfd -> crohme_normalizer -> m4",
                }
            )
            markdown_parts.append(f"$$\n{m4_latex}\n$$")

        annotated_cv = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        for index, item in enumerate(elements):
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated_cv, (x1, y1), (x2, y2), (180, 0, 180), 3)
            cv2.putText(
                annotated_cv,
                f"#{index + 1} M4 + P2T MFD",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 0, 180),
                2,
            )
        success, annotated_buffer = cv2.imencode(".png", annotated_cv)
        if not success:
            raise RuntimeError("Không thể tạo ảnh đánh dấu vùng công thức.")
        annotated_uri = (
            "data:image/png;base64,"
            + base64.b64encode(annotated_buffer).decode("utf-8")
        )

        return {
            "status": "success",
            "engine": "m4",
            "doc_id": str(uuid.uuid4()),
            "filename": filename,
            "width": width,
            "height": height,
            "annotated_image": annotated_uri,
            "elements": elements,
            "markdown": "\n\n".join(markdown_parts),
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Mô hình M4 không thể nhận dạng ảnh: {exc}",
        ) from exc

if __name__ == "__main__":
    import uvicorn
    # Không dùng reload vì tiến trình reloader sẽ nạp thêm một bản mô hình,
    # làm tăng mạnh mức sử dụng và phân mảnh bộ nhớ GPU.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

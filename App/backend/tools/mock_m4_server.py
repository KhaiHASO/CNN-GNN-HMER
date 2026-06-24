from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Mock M4 Server")


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    return {
        "latex": r"(a+b)^n=\sum_{k=0}^{n} C_n^k a^{n-k}b^k",
        "confidence": 0.99,
    }


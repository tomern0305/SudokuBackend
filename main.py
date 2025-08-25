# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, conlist
from pathlib import Path
from datetime import datetime
import torch
import logging

from Solver import solve_vector
from ocr.pipeline import image_bytes_to_vector
from ocr.utils import initModel


# -------- FastAPI setup --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# -------- types --------
Vector81 = conlist(int, min_length=81, max_length=81)
class SolveRequest(BaseModel):
    vector: Vector81
class SolveResponse(BaseModel):
    vector: Vector81

# -------- model loaded once --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None

@app.on_event("startup")
def _load_model_once():
    global MODEL
    try:
        MODEL = initModel(path="mnist_cnn.pth", device=DEVICE)
        logging.info("Model loaded on %s", DEVICE)
    except Exception as e:
        logging.exception("Failed to load model: %s", e)
        # keep going so /solve works; /scan will 500 until fixed

# -------- routes --------
@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    try:
        solved = solve_vector(req.vector)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"vector": solved}

@app.post("/scan")
async def scan(request: Request, image: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(500, "Model not loaded on server startup")

    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(400, "Please upload a JPG or PNG")

    # save uploaded ROI (debug)
    ext = ".jpg" if image.content_type == "image/jpeg" else ".png"
    fname = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ext
    dest = UPLOAD_DIR / fname
    body = await image.read()
    dest.write_bytes(body)

    try:
        vec = image_bytes_to_vector(body, MODEL, DEVICE, conf_thr=0.90)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("scan failed: %s", e)
        raise HTTPException(400, f"Failed to parse grid: {e}")

    base = str(request.base_url).rstrip("/")
    return {"vector": [int(x) for x in vec], "image_url": f"{base}/uploads/{fname}"}

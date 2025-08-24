# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, conlist
from pathlib import Path
from datetime import datetime

from Solver import solve_vector

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

Vector81 = conlist(int, min_length=81, max_length=81)

class SolveRequest(BaseModel):
    vector: Vector81

class SolveResponse(BaseModel):
    vector: Vector81

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    try:
        solved = solve_vector(req.vector)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"vector": solved}

@app.post("/scan")
async def scan(request: Request, image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(400, "Please upload a JPG or PNG")

    # save uploaded ROI to ./uploads with a timestamped name
    ext = ".jpg" if image.content_type == "image/jpeg" else ".png"
    filename = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ext
    dest = UPLOAD_DIR / filename
    dest.write_bytes(await image.read())

    # absolute URL you can open from any device on the LAN
    base = str(request.base_url).rstrip("/")  # e.g. http://192.168.1.23:8000
    image_url = f"{base}/uploads/{filename}"

    # TODO: run image->vector; stub zeros for now
    return {"vector": [0] * 81, "image_url": image_url}

# ocr/pipeline.py
import numpy as np
import cv2
from fastapi import HTTPException
from ocr.utils import preProcess, biggestContour, reorder, splitBoxes, getPrediction

def _read_bgr(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img

def _find_and_warp_board(bgr):
    proc = preProcess(bgr)
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest, maxArea = biggestContour(contours)
    if biggest.size == 0:
        # If client already sent a tight ROI, just use it as-is (square warp)
        h, w = bgr.shape[:2]
        side = min(w, h)
        # center-crop square then continue
        left = (w - side) // 2
        top  = (h - side) // 2
        bgr = bgr[top:top+side, left:left+side].copy()
        return cv2.resize(bgr, (450, 450), interpolation=cv2.INTER_AREA)

    # perspective warp to a 450x450 board
    biggest = reorder(biggest).reshape(4, 2).astype(np.float32)  # tl,tr,bl,br
    dst = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    M = cv2.getPerspectiveTransform(biggest, dst)
    warp = cv2.warpPerspective(bgr, M, (450, 450))
    return warp

def image_bytes_to_vector(image_bytes, model, device, conf_thr=0.80):
    """
    Returns a Python list of length 81 with ints 0..9 (0 = blank).
    Raises HTTPException(400, ...) on parsing failure.
    """
    bgr = _read_bgr(image_bytes)
    board = _find_and_warp_board(bgr)

    # threshold for clean cells, then split 9x9
    thr = preProcess(board)
    boxes = splitBoxes(thr)
    if len(boxes) != 81:
        raise HTTPException(400, f"Expected 81 cells, got {len(boxes)}")

    preds, conf = getPrediction(boxes, model, device, blank_by_conf=True, conf_thr=conf_thr)

    vec = preds.tolist()
    if len(vec) != 81:
        raise HTTPException(500, "Prediction did not return 81 outputs")
    # ensure ints and bounds
    vec = [int(x) if 0 <= int(x) <= 9 else 0 for x in vec]
    return vec

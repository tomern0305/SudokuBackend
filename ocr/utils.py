# ocr/utils.py
import cv2
import numpy as np
import torch
from modelEmnist import SmallCNN

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def initModel(path="mnist_cnn.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def preProcess(image):
    # BGR -> gray -> blur -> adaptive threshold (white background)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(imageGray, (5, 5), 1)
    imageThreshold = cv2.adaptiveThreshold(imageBlur, 255, 1, 1, 11, 2)
    return imageThreshold

def biggestContour(contours):
    biggest = np.array([])
    maxArea = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxArea:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest, maxArea

def reorder(biggest):
    # input: (4,1,2) -> output: (4,1,2) order: tl, tr, bl, br
    biggest = biggest.reshape(4, 2)
    biggestNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    biggestNew[0] = biggest[np.argmin(add)]  # tl
    biggestNew[3] = biggest[np.argmax(add)]  # br
    diff = np.diff(biggest, axis=1)
    biggestNew[1] = biggest[np.argmin(diff)]  # tr
    biggestNew[2] = biggest[np.argmax(diff)]  # bl
    return biggestNew

def splitBoxes(image):
    rows = np.vsplit(image, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes

def getPrediction(boxes, model, device, blank_by_conf=True, conf_thr=0.8):
    model.eval()
    imgs = []
    for cell in boxes:
        if cell.ndim == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        cell = cell.astype(np.uint8)

        # MNIST/EMNIST expects white digit on black; invert if needed
        if cell.mean() > 127:
            cell = 255 - cell

        cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(cell).float().div(255.0).unsqueeze(0)
        t = (t - MNIST_MEAN) / MNIST_STD
        imgs.append(t)

    x = torch.stack(imgs, dim=0).to(device)  # [81,1,28,28]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

    preds = preds.cpu().numpy()   # 0..9 (assuming model outputs 10 classes)
    conf  = conf.cpu().numpy()

    if blank_by_conf:
        preds = preds.copy()
        preds[conf < conf_thr] = 0

    return preds, conf

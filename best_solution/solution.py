import numpy as np
from typing import List, Union, Tuple
from ultralytics import YOLO
import torch
import os
import cv2

model_path = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(model_path)

def im_spl_with_overlap(img: np.ndarray, p_sz: int = 1280, overlap: int = 256) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Разбивает изображение на патчи с перекрытием."""
    H, W = img.shape[:2]
    pts = []
    stride = p_sz - overlap

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + p_sz, H)
            x_end = min(x + p_sz, W)
            y_start = max(0, y_end - p_sz)
            x_start = max(0, x_end - p_sz)
            pt = img[y_start:y_end, x_start:x_end]
            pts.append((pt, (x_start, y_start)))

    return pts


def nms_bboxes(bboxes: List[dict], iou_threshold: float = 0.45) -> List[dict]:
    """Применяет NMS к списку НОРМАЛИЗОВАННЫХ боксов (xc, yc, w, h в [0,1])."""
    if len(bboxes) == 0:
        return []

    boxes = np.array([[b['xc'] - b['w']/2, b['yc'] - b['h']/2, b['xc'] + b['w']/2, b['yc'] + b['h']/2] for b in bboxes])
    scores = np.array([b['score'] for b in bboxes])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        union = np.maximum(union, 1e-9)  # защита от деления на 0

        ovr = inter / union
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return [bboxes[i] for i in keep]


# ---------- клиппинг нормализованных боксов ----------
def _clip_bbox_norm(b: dict, eps: float = 1e-9) -> Union[dict, None]:
    """
    Режет бокс по границам изображения (нормализованные координаты).
    На входе: dict с полями xc, yc, w, h, score, label (все в [0,1], кроме score/label).
    Возвращает обрезанный бокс или None, если стал вырожденным.
    """
    x1 = max(0.0, b['xc'] - b['w'] / 2.0)
    y1 = max(0.0, b['yc'] - b['h'] / 2.0)
    x2 = min(1.0, b['xc'] + b['w'] / 2.0)
    y2 = min(1.0, b['yc'] + b['h'] / 2.0)

    w = x2 - x1
    h = y2 - y1
    if w <= eps or h <= eps:
        return None

    return {
        'xc': (x1 + x2) / 2.0,
        'yc': (y1 + y2) / 2.0,
        'w': w,
        'h': h,
        'label': b.get('label', 0),
        'score': b['score'],
    }


def clip_bboxes_to_image(bboxes: List[dict]) -> List[dict]:
    """Обрезает список нормализованных боксов до границ изображения [0,1]."""
    out = []
    for b in bboxes:
        cb = _clip_bbox_norm(b)
        if cb is not None:
            out.append(cb)
    return out
# ----------------------------------------------------


def infer_image_bbox(img: np.ndarray) -> List[dict]:
    """Детектирует объекты и возвращает список НОРМАЛИЗОВАННЫХ боксов с клиппингом и NMS."""
    H, W = img.shape[:2]

    pts = im_spl_with_overlap(img)
    all_bxs = []

    for pt, (x_off, y_off) in pts:
        res = model.predict(
            source=pt,
            imgsz=1536,
            conf=0.20,       # ВЕРНУЛИ старый порог уверенности
            iou=0.45,        # ВЕРНУЛИ старый IoU для внутреннего NMS
            max_det=30,
            device=0 if torch.cuda.is_available() else 'cpu'
        )

        for r in res:
            for box in r.boxes:
                xc_p = box.xywhn[0][0].item()
                yc_p = box.xywhn[0][1].item()
                w_p  = box.xywhn[0][2].item()
                h_p  = box.xywhn[0][3].item()
                cf   = box.conf[0].item()

                # Перевод в НОРМАЛИЗОВАННЫЕ координаты целого изображения
                xc = (xc_p * pt.shape[1] + x_off) / W
                yc = (yc_p * pt.shape[0] + y_off) / H
                w_n = (w_p * pt.shape[1]) / W
                h_n = (h_p * pt.shape[0]) / H

                all_bxs.append({
                    'xc': xc,
                    'yc': yc,
                    'w': w_n,
                    'h': h_n,
                    'label': 0,
                    'score': cf
                })

    # Сначала клиппим боксы, затем делаем наш NMS
    all_bxs = clip_bboxes_to_image(all_bxs)
    final_bboxes = nms_bboxes(all_bxs)
    return final_bboxes


def predict(imgs: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Обёртка над infer_image_bbox для списка изображений или одного изображения."""
    results = []
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    for img in imgs:
        img_rslts = infer_image_bbox(img)
        results.append(img_rslts)

    return results

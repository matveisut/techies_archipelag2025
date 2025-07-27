import numpy as np
from typing import List, Union, Tuple
from ultralytics import YOLO
import torch

import os

model_path = os.path.join(os.path.dirname(__file__), "yolo11x.pt")
model = YOLO(model_path)


def im_spl(img: np.ndarray, p_sz: int = 640) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Разбивает изображение на патчи 640x640 с координатами исходного изображения"""
    H, W = img.shape[:2]
    pts = []
    for y in range(0, H, p_sz):
        for x in range(0, W, p_sz):
            pt = img[y:y+p_sz, x:x+p_sz]
            pts.append((pt, (x, y)))
    return pts

def box_mrg(bxs: List[dict], im_sz: Tuple[int, int]) -> List[dict]:
    """Объединяет пересекающиеся боксы и нормализует координаты к исходному изображению"""
    if not bxs:
        return []
    
    abs_bxs = []
    W_im, H_im = im_sz
    
    for bx in bxs:
        xc_abs = bx['xc'] * W_im
        yc_abs = bx['yc'] * H_im
        w_abs = bx['w'] * W_im
        h_abs = bx['h'] * H_im
        
        abs_bxs.append({
            'xc': xc_abs,
            'yc': yc_abs,
            'w': w_abs,
            'h': h_abs,
            'label': bx['label'],
            'score': bx['score']
        })
    
    mrgd = []
    while abs_bxs:
        curr = abs_bxs.pop(0)
        to_mrg = [curr]
        
        i = 0
        while i < len(abs_bxs):
            bx = abs_bxs[i]
            if bxs_intrsct(curr, bx):
                to_mrg.append(abs_bxs.pop(i))
            else:
                i += 1
        
        if len(to_mrg) > 1:
            mrgd_bx = cmn_bxs(to_mrg)
            mrgd.append(mrgd_bx)
        else:
            mrgd.append(curr)
    
    fin_bxs = []
    for bx in mrgd:
        fin_bxs.append({
            'xc': bx['xc'] / W_im,
            'yc': bx['yc'] / H_im,
            'w': bx['w'] / W_im,
            'h': bx['h'] / H_im,
            'label': bx['label'],
            'score': max(b['score'] for b in to_mrg)
        })
    
    return fin_bxs

def bxs_intrsct(bx1: dict, bx2: dict) -> bool:
    """Проверяет пересекаются ли два бокса"""
    x1_1 = bx1['xc'] - bx1['w']/2
    y1_1 = bx1['yc'] - bx1['h']/2
    x2_1 = bx1['xc'] + bx1['w']/2
    y2_1 = bx1['yc'] + bx1['h']/2
    
    x1_2 = bx2['xc'] - bx2['w']/2
    y1_2 = bx2['yc'] - bx2['h']/2
    x2_2 = bx2['xc'] + bx2['w']/2
    y2_2 = bx2['yc'] + bx2['h']/2
    
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

def cmn_bxs(bxs: List[dict]) -> dict:
    """Объединяет несколько боксов в один минимальный охватывающий прямоугольник"""
    x_c = []
    y_c = []
    scrs = []
    
    for bx in bxs:
        x1 = bx['xc'] - bx['w']/2
        x2 = bx['xc'] + bx['w']/2
        y1 = bx['yc'] - bx['h']/2
        y2 = bx['yc'] + bx['h']/2
        
        x_c.extend([x1, x2])
        y_c.extend([y1, y2])
        scrs.append(bx['score'])
    
    x_min, x_max = min(x_c), max(x_c)
    y_min, y_max = min(y_c), max(y_c)
    
    return {
        'xc': (x_min + x_max) / 2,
        'yc': (y_min + y_max) / 2,
        'w': x_max - x_min,
        'h': y_max - y_min,
        'label': bxs[0]['label'],
        'score': max(scrs)
    }

def infer_image_bbox(img: np.ndarray) -> List[dict]:
    H, W = img.shape[:2]
    pts = im_spl(img)
    all_bxs = []
    
    for pt, (x_off, y_off) in pts:
        res = model.predict(source=pt, imgsz=640, device=0 if torch.cuda.is_available() else 'cpu')
        
        for r in res:
            for b in r.boxes:
                xc_p = b.xywhn[0][0].item()
                yc_p = b.xywhn[0][1].item()
                w_p = b.xywhn[0][2].item()
                h_p = b.xywhn[0][3].item()
                cf = b.conf[0].item()
                
                xc = (xc_p * pt.shape[1] + x_off) / W
                yc = (yc_p * pt.shape[0] + y_off) / H
                w_n = w_p * pt.shape[1] / W
                h_n = h_p * pt.shape[0] / H
                
                all_bxs.append({
                    'xc': xc,
                    'yc': yc,
                    'w': w_n,
                    'h': h_n,
                    'label': 0,
                    'score': cf
                })
    
    mrgd_bxs = box_mrg(all_bxs, (W, H))
    
    return mrgd_bxs

def predict(imgs: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    results = []
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    for img in imgs:        
        img_rslts = infer_image_bbox(img)
        results.append(img_rslts)
    
    return results
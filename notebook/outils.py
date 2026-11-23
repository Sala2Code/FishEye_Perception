"""
outils.py — Chargement des images, détection YOLO26 sur chaque représentation, comparaison.

Toutes les détections sont ramenées dans l'image fisheye d'origine (via les rayons), ce qui
permet de les comparer entre représentations et, si elles existent, aux annotations.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from projections import box_iou, nms, transfer_box

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------------------
# FishEye8K (Hugging Face, format FiftyOne : samples.json + data/)
# --------------------------------------------------------------------------------------
def read_fisheye8k_index(samples_json):
    """Liste des échantillons : chemin relatif, caméra, découpage, boîtes relatives [x,y,w,h]."""
    with open(samples_json) as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data
    out = []
    for s in samples:
        fp = s["filepath"]
        rel = fp[fp.index("data/"):] if "data/" in fp else "data/" + Path(fp).name
        dets = (s.get("detections") or {}).get("detections", [])
        name = Path(rel).name
        loc = s.get("location") or name.split("_")[0]
        out.append(dict(rel=rel, name=name, location=str(loc), split=",".join(s.get("tags", [])),
                        gt=[(d.get("label"), d.get("bounding_box")) for d in dets]))
    return out


def select_samples(index, per_location=8, label="Pedestrian", min_count=1, seed=0):
    """Tire `per_location` images par caméra, parmi celles qui contiennent au moins
    `min_count` objets de la classe `label` (couvre toutes les caméras, toutes les heures)."""
    by_loc = defaultdict(list)
    for s in index:
        if sum(l == label for l, _ in s["gt"]) >= min_count:
            by_loc[s["location"]].append(s)
    rng = random.Random(seed)
    chosen = []
    for loc in sorted(by_loc):
        lst = by_loc[loc][:]
        rng.shuffle(lst)
        chosen += lst[:per_location]
    return chosen


def gt_pixels(sample, W, H, label="Pedestrian"):
    """Boîtes annotées de la classe `label`, en pixels (x1, y1, x2, y2)."""
    out = []
    for l, b in sample.get("gt", []):
        if l == label and b:
            x, y, w, h = b
            out.append([x * W, y * H, (x + w) * W, (y + h) * H])
    return np.array(out, float).reshape(-1, 4)


def list_folder(folder):
    """Mode « dossier » : toutes les images d'un dossier (ex. captures de votre banc)."""
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
    return [dict(path=p, name=p.name, location=p.parent.name, gt=[]) for p in files]


# --------------------------------------------------------------------------------------
# Détection
# --------------------------------------------------------------------------------------
def imgsz_for(img):
    """Taille d'entrée YOLO = plus grand côté arrondi à 32 : l'image n'est pas réduite."""
    return int(math.ceil(max(img.shape[:2]) / 32) * 32)


def detect(model, img, conf=0.01, device=None):
    r = model.predict(img, classes=[0], conf=conf, imgsz=imgsz_for(img), max_det=300,
                      verbose=False, device=device)[0]
    return (r.boxes.xyxy.cpu().numpy().astype(float), r.boxes.conf.cpu().numpy().astype(float),
            float(sum(r.speed.values())))


def run_representation(rep, image, model, conf=0.01, device=None, keep_images=False):
    """Remappe, détecte, puis ramène les boîtes dans l'image fisheye d'origine.
    Pour le multivues, les doublons entre vues sont supprimés par NMS dans l'image d'origine."""
    fe = rep["fisheye"]
    out = dict(t_remap=0.0, t_infer=0.0, t_fuse=0.0, views=[])
    B, S = [], []
    for target, lut in rep["views"]:
        is_raw = target is fe
        t0 = time.perf_counter()
        im = image if is_raw else lut.remap(image)
        if not is_raw:
            out["t_remap"] += (time.perf_counter() - t0) * 1e3
        boxes, scores, t = detect(model, im, conf, device)
        out["t_infer"] += t
        out["views"].append(dict(image=im if keep_images else None, boxes=boxes, scores=scores))
        t0 = time.perf_counter()
        for b, s in zip(boxes, scores):
            tb = b if is_raw else transfer_box(b, target, fe)
            if tb is not None and tb[2] - tb[0] > 1 and tb[3] - tb[1] > 1:
                B.append(tb); S.append(s)
        out["t_fuse"] += (time.perf_counter() - t0) * 1e3
    B, S = np.array(B).reshape(-1, 4), np.array(S)
    if rep.get("dedup") and len(B):
        t0 = time.perf_counter()
        k = nms(B, S, 0.5)
        B, S = B[k], S[k]
        out["t_fuse"] += (time.perf_counter() - t0) * 1e3
    out["boxes_fe"], out["scores_fe"] = B, S
    return out


def match(det_boxes, det_scores, gt_boxes, iou_thr):
    """Appariement glouton par score décroissant ; renvoie, pour chaque boîte annotée,
    True si une détection la retrouve avec IoU >= iou_thr."""
    found = np.zeros(len(gt_boxes), bool)
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return found
    ious = box_iou(det_boxes, gt_boxes)
    for d in np.argsort(-det_scores):
        cand = np.where(found, -1, ious[d])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            found[j] = True
    return found


def draw(img, boxes, color, scores=None, thick=2):
    out = img.copy()
    for i, b in enumerate(np.asarray(boxes).reshape(-1, 4)):
        p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
        cv2.rectangle(out, p1, p2, color, thick)
        if scores is not None:
            cv2.putText(out, f"{scores[i]:.2f}", (p1[0], max(p1[1] - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out

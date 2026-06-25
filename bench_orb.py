"""
Banco ISOLATO per valutare ORB (feature matching) vs pHash sulle carte olografiche.
NON tocca la pipeline. Solo misura.

Per ogni cella di carte4.jpg (nome già noto), confronta la cella contro le immagini
TCGdex a risoluzione piena (/high.png) di TUTTI i candidati di quel nome:
 - ORB: numero di inlier RANSAC (match geometricamente coerenti) = "buoni match"
 - pHash: distanza (per confronto col metodo attuale)
Stampa, per cella, i candidati ordinati per inlier ORB, con id/localId/pHash, così si
vede se ORB STACCA il candidato giusto dagli altri (dove il pHash falliva).

Uso:  python bench_orb.py
"""

import os
import cv2
import numpy as np
import requests
import imagehash
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from funzioni import cerca_carta_api
from bench_rifinitura import celle_da_bande, bande_uniformi_e_reali, _phash_bgr

CELLE_NOMI = {
    (0, 0): "Pikachu", (0, 1): "Pikachu",   (0, 2): "Martes",
    (1, 0): "Articuno", (1, 1): "Flareon",  (1, 2): "Raichu",
    (2, 0): "Drampa",   (2, 1): "Vaporeon", (2, 2): "Exeggutor",
}
CACHE_IMG = "orb_cache"          # immagini full-res dei candidati
MAX_CAND = 25                    # candidati per nome (cap)
NFEAT = 2000                     # feature ORB


def _scarica_full(card):
    # Scarica l'immagine /high.png del candidato in cache. Ritorna percorso o None.
    cid = card["id"]
    path = os.path.join(CACHE_IMG, f"{cid}.png")
    if os.path.exists(path):
        return path
    url = card.get("image")
    if not url:
        return None
    for suff in ("/high.png", "/low.png"):
        try:
            r = requests.get(url + suff, timeout=20)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
        except requests.exceptions.RequestException:
            continue
    return None


def _grigia_resize(img_bgr, alt=700):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if h != alt:
        g = cv2.resize(g, (int(w * alt / h), alt), interpolation=cv2.INTER_AREA)
    return g


def orb_inliers(g1, g2):
    # ORB + ratio test + RANSAC. Ritorna (n_buoni_ratio, n_inlier_ransac).
    orb = cv2.ORB_create(NFEAT)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for mn in matches:
        if len(mn) == 2 and mn[0].distance < 0.75 * mn[1].distance:
            good.append(mn[0])
    if len(good) < 4:
        return len(good), 0
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inl = int(mask.sum()) if mask is not None else 0
    return len(good), inl


def main():
    os.makedirs(CACHE_IMG, exist_ok=True)
    (_, _), (re_r, re_c) = bande_uniformi_e_reali()
    celle = {(r, c): cell for (r, c, nome, cell) in celle_da_bande(re_r, re_c)}

    for (r, c), nome in CELLE_NOMI.items():
        cell = celle[(r, c)]
        g_cell = _grigia_resize(cell)
        ph_cell = _phash_bgr(cell)

        candidati = (cerca_carta_api(nome) or [])[:MAX_CAND]
        # scarica le immagini full-res in parallelo
        with ThreadPoolExecutor(max_workers=8) as ex:
            paths = list(ex.map(_scarica_full, candidati))

        righe = []
        for card, path in zip(candidati, paths):
            if not path:
                continue
            ref = cv2.imread(path)
            if ref is None:
                continue
            _, inl = orb_inliers(g_cell, _grigia_resize(ref))
            try:
                ph = ph_cell - imagehash.phash(Image.open(path))
            except OSError:
                ph = None
            righe.append((inl, ph, card["id"], str(card.get("localId", ""))))

        righe.sort(key=lambda x: -x[0])   # per inlier ORB decrescente
        print(f"\n=== cella {r}_{c}  '{nome}'  ({len(righe)} candidati) ===")
        print(f"  {'ORB':>4} {'pHash':>6}  {'id':22} localId")
        for inl, ph, cid, lid in righe[:6]:
            print(f"  {inl:>4} {str(ph):>6}  {cid:22} {lid}")


if __name__ == "__main__":
    main()

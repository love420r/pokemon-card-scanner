"""
Banco di misura per il ritaglio delle celle album (carte4.jpg).

Per ogni cella conosciamo il nome del Pokémon. La METRICA è la distanza pHash
MINIMA tra il ritaglio e i candidati TCGdex di quel nome: un ritaglio ben
inquadrato (che segue i bordi veri della carta) abbassa la distanza.

I pHash di riferimento si scaricano UNA volta e si mettono in cache
(bench_ref_cache.json), così le strategie si testano offline e veloci.

Uso:  python bench_rifinitura.py   -> confronta divisione UNIFORME vs BANDE REALI
"""

import os
import json
import cv2
import requests
import imagehash
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from funzioni import (cerca_carta_api, carica_e_ridimensiona, rileva_griglia,
                      _concilia_bande, rifinisci_cella)

FOTO = "carte4.jpg"
CACHE = "bench_ref_cache.json"

# (riga, colonna, nome base per la ricerca TCGdex).
CELLE = [
    (0, 0, "Pikachu"), (0, 1, "Pikachu"),   (0, 2, "Martes"),
    (1, 0, "Articuno"), (1, 1, "Flareon"),  (1, 2, "Raichu"),
    (2, 0, "Drampa"),   (2, 1, "Vaporeon"), (2, 2, "Exeggutor"),
]
NOME_DI = {(r, c): nm for (r, c, nm) in CELLE}


def celle_da_bande(bande_righe, bande_colonne):
    # Ritaglia le 9 celle dalla foto full-res usando bande in scala ridotta (1000px).
    orig = cv2.imread(FOTO)
    rap = orig.shape[1] / 1000.0
    out = []
    for r, (y0, y1) in enumerate(bande_righe):
        for c, (x0, x1) in enumerate(bande_colonne):
            cell = orig[int(y0 * rap):int(y1 * rap), int(x0 * rap):int(x1 * rap)]
            out.append((r, c, NOME_DI.get((r, c), "?"), cell))
    return out


def _phash_url(url_base):
    try:
        rsp = requests.get(url_base + "/low.png", timeout=15)
        if rsp.status_code != 200:
            return None
        return str(imagehash.phash(Image.open(BytesIO(rsp.content))))
    except (requests.exceptions.RequestException, OSError):
        return None


def costruisci_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    nomi = sorted({nome for _, _, nome in CELLE})
    cache = {}
    for nome in nomi:
        candidati = cerca_carta_api(nome) or []
        urls = [c["image"] for c in candidati if c.get("image")][:60]
        hashes = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for h in ex.map(_phash_url, urls):
                if h:
                    hashes.append(h)
        cache[nome] = hashes
        print(f"  '{nome}': {len(hashes)} riferimenti")
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return cache


def _phash_bgr(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def valuta(etichetta, bande_righe, bande_colonne, cache, crop_fn=rifinisci_cella):
    # Ritaglia le celle dalle bande date, applica crop_fn, misura la distanza pHash
    # minima ai riferimenti. Ritorna (distanze, media).
    print(f"\n=== {etichetta} ===")
    print(f"  bande righe: {bande_righe}")
    distanze = []
    for (r, c, nome, cell) in celle_da_bande(bande_righe, bande_colonne):
        crop = crop_fn(cell) if crop_fn else cell
        if crop is None or crop.size == 0:
            distanze.append(None)
            continue
        ph = _phash_bgr(crop)
        refs = cache.get(nome, [])
        if not refs:
            distanze.append(None)
            continue
        d = min(ph - imagehash.hex_to_hash(h) for h in refs)
        flag = "  <13 OK" if d < 13 else ""
        print(f"  cella {r}_{c} {nome:12} dist={d}{flag}")
        distanze.append(d)
    valide = [d for d in distanze if d is not None]
    media = sum(valide) / len(valide) if valide else 0
    sotto = sum(1 for d in valide if d < 13)
    print(f"  --> media={media:.1f}  |  sotto 13: {sotto}/{len(valide)}")
    return distanze, media


def bande_uniformi_e_reali():
    # Calcola le bande "PRIMA" (divisione uniforme) e "DOPO" (bande reali conciliate
    # a 3×3) per carte4.jpg, in scala ridotta 1000px.
    img = carica_e_ridimensiona(FOTO, 1000)
    H, W = img.shape[0], img.shape[1]
    br, bc = rileva_griglia(img)
    # DOPO: bande reali conciliate a 3 righe / 3 colonne
    reali_r = _concilia_bande(br, 3, None, H)
    reali_c = _concilia_bande(bc, 3, None, W)
    # PRIMA: divisione uniforme dell'estensione (vecchio comportamento)
    er = (br[0][0], br[-1][1]); passo = (er[1] - er[0]) / 3
    uni_r = [(int(er[0] + k * passo), int(er[0] + (k + 1) * passo)) for k in range(3)]
    ec = (bc[0][0], bc[-1][1]); passoc = (ec[1] - ec[0]) / 3
    uni_c = [(int(ec[0] + k * passoc), int(ec[0] + (k + 1) * passoc)) for k in range(3)]
    return (uni_r, uni_c), (reali_r, reali_c)


def _rifinisci_vecchia(cella):
    # La rifinitura della pipeline PRECEDENTE: raddrizza + taglia bande + normalizza
    # al rapporto carta (ratio-crop centrato). Serve solo qui per il confronto onesto.
    from funzioni import _raddrizza_cella, _taglia_bande_scure
    try:
        c = _raddrizza_cella(cella)
        xl, yt, xr, yb = _taglia_bande_scure(c)
        sub = c[yt:yb, xl:xr]
        if sub.size == 0:
            return cella
        h, w = sub.shape[:2]
        if h / w > 1.34:
            nh = int(w * 1.34); y = (h - nh) // 2; sub = sub[y:y + nh, :]
        else:
            nw = int(h / 1.34); x = (w - nw) // 2; sub = sub[:, x:x + nw]
        return sub if sub.size else cella
    except Exception:
        return cella


if __name__ == "__main__":
    cache = costruisci_cache()
    (uni_r, uni_c), (reali_r, reali_c) = bande_uniformi_e_reali()
    d0, m0 = valuta("PRIMA — UNIFORME + rifinisci vecchia (ratio-crop)",
                    uni_r, uni_c, cache, crop_fn=_rifinisci_vecchia)
    d1, m1 = valuta("DOPO — BANDE REALI estese + rifinisci leggera",
                    reali_r, reali_c, cache)   # crop_fn = rifinisci_cella (nuova)
    print(f"\nMEDIA pHash:  prima {m0:.1f}  ->  dopo {m1:.1f}")
    s0 = sum(1 for d in d0 if d is not None and d < 13)
    s1 = sum(1 for d in d1 if d is not None and d < 13)
    print(f"Celle <13:    prima {s0}/9  ->  dopo {s1}/9")

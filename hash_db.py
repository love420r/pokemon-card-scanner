"""
Database locale di pHash delle carte Pokémon.

Idea: invece di scaricare le immagini dei candidati a ogni scansione (lento e
fragile), si costruisce UNA volta un indice locale {id, nome, set, numero, phash}.
Poi riconoscere una carta = calcolare il suo pHash e cercare il più vicino
nell'indice, tutto in locale.

Uso tipico:
    import hash_db
    hash_db.costruisci(sets=["swsh3", "sv03"])      # build (una tantum, resumabile)
    indice = hash_db.carica_indice()                # carica in memoria
    risultati = hash_db.cerca(phash_carta, indice)  # top match
"""

import sqlite3
import requests
import numpy as np
import imagehash
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = "hash_db.sqlite"
BASE_API = "https://api.tcgdex.net/v2"

# Tabella di popcount per byte: POPCOUNT8[b] = numero di bit a 1 in b (0-255).
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# COSTRUZIONE
# ---------------------------------------------------------------------------

def _connessione(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS carte (
            id        TEXT PRIMARY KEY,
            nome      TEXT,
            set_id    TEXT,
            set_nome  TEXT,
            local_id  TEXT,
            phash     TEXT
        )
    """)
    return con


def _lista_set_id(lingua):
    # Tutti gli id dei set disponibili su TCGdex.
    r = requests.get(f"{BASE_API}/{lingua}/sets", timeout=15)
    r.raise_for_status()
    return [s["id"] for s in r.json()]


def _carte_di_un_set(set_id, lingua):
    # Ritorna (set_nome, [brief_carte]). Ogni brief ha id, localId, name, image.
    r = requests.get(f"{BASE_API}/{lingua}/sets/{set_id}", timeout=15)
    r.raise_for_status()
    dati = r.json()
    return dati.get("name", set_id), dati.get("cards", [])


def _scarica_phash(url_immagine):
    # Scarica low.png e calcola il pHash. Ritorna stringa hex o None.
    try:
        r = requests.get(url_immagine + "/low.png", timeout=15)
        if r.status_code != 200:
            return None
        h = imagehash.phash(Image.open(BytesIO(r.content)))
        return str(h)
    except (requests.exceptions.RequestException, OSError):
        return None


def costruisci(sets=None, lingua="en", db_path=DB_PATH, workers=12):
    """
    Costruisce/aggiorna l'indice locale.

    sets   = lista di set_id da includere (es. ["swsh3", "sv03"]).
             None = TUTTI i set (pesante: decine di migliaia di immagini).
    lingua = lingua dell'API. "en" dà il catalogo più completo; l'artwork è
             identico tra le lingue, quindi il pHash matcha anche carte ITA.
    Resumabile: gli id già presenti nel DB vengono saltati.
    """
    con = _connessione(db_path)
    gia_presenti = {r[0] for r in con.execute("SELECT id FROM carte")}

    if sets is None:
        print("Scarico la lista di TUTTI i set...")
        sets = _lista_set_id(lingua)
    print(f"Set da elaborare: {len(sets)}. Già nel DB: {len(gia_presenti)} carte.\n")

    totale_nuove = 0
    for i, set_id in enumerate(sets, 1):
        try:
            set_nome, briefs = _carte_di_un_set(set_id, lingua)
        except requests.exceptions.RequestException:
            print(f"  [{i}/{len(sets)}] {set_id}: errore rete, salto")
            continue

        # Tieni solo le carte con immagine e non ancora indicizzate.
        da_fare = [b for b in briefs if b.get("image") and b["id"] not in gia_presenti]
        if not da_fare:
            print(f"  [{i}/{len(sets)}] {set_id} ({set_nome}): niente di nuovo")
            continue

        nuove = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futuri = {ex.submit(_scarica_phash, b["image"]): b for b in da_fare}
            for fut in as_completed(futuri):
                b = futuri[fut]
                ph = fut.result()
                if ph is None:
                    continue
                nuove.append((b["id"], b.get("name", ""), set_id, set_nome,
                              str(b.get("localId", "")), ph))

        con.executemany("INSERT OR REPLACE INTO carte VALUES (?,?,?,?,?,?)", nuove)
        con.commit()
        totale_nuove += len(nuove)
        print(f"  [{i}/{len(sets)}] {set_id} ({set_nome}): +{len(nuove)} carte")

    con.close()
    print(f"\nFatto. Carte nuove indicizzate: {totale_nuove}")


# ---------------------------------------------------------------------------
# CARICAMENTO E RICERCA
# ---------------------------------------------------------------------------

class Indice:
    # Indice in memoria per ricerca veloce. Gli array sono allineati per riga.
    def __init__(self, ids, nomi, set_ids, local_ids, phash_uint64):
        self.ids = ids
        self.nomi = nomi
        self.set_ids = set_ids
        self.local_ids = local_ids
        self.phash = phash_uint64        # np.ndarray uint64, shape (N,)

    def __len__(self):
        return len(self.ids)


def carica_indice(db_path=DB_PATH):
    """Carica tutto il DB in memoria. Ritorna un Indice pronto per cerca()."""
    con = sqlite3.connect(db_path)
    righe = con.execute(
        "SELECT id, nome, set_id, set_nome, local_id, phash FROM carte"
    ).fetchall()
    con.close()

    ids, nomi, set_ids, local_ids, phs = [], [], [], [], []
    for _id, nome, set_id, _set_nome, local_id, ph in righe:
        ids.append(_id)
        nomi.append(nome)
        set_ids.append(set_id)
        local_ids.append(local_id)
        phs.append(np.uint64(int(ph, 16)))     # hex -> uint64

    arr = np.array(phs, dtype=np.uint64) if phs else np.empty(0, dtype=np.uint64)
    return Indice(ids, nomi, set_ids, local_ids, arr)


def _phash_a_uint64(phash):
    # Accetta un imagehash.ImageHash o una stringa hex; ritorna uint64.
    return np.uint64(int(str(phash), 16))


def cerca(phash_query, indice, top_n=5):
    """
    Cerca le carte più vicine a phash_query nell'indice.
    phash_query = imagehash.ImageHash (o stringa hex).
    Ritorna lista di dict ordinati per distanza crescente:
        {distanza, id, nome, set_id, local_id}
    """
    if len(indice) == 0:
        return []

    q = _phash_a_uint64(phash_query)
    xor = indice.phash ^ q                       # (N,) uint64
    byte_view = xor.view(np.uint8).reshape(-1, 8)
    distanze = POPCOUNT8[byte_view].sum(axis=1)  # (N,) distanza di Hamming

    ordine = np.argsort(distanze)[:top_n]
    out = []
    for j in ordine:
        out.append({
            "distanza": int(distanze[j]),
            "id": indice.ids[j],
            "nome": indice.nomi[j],
            "set_id": indice.set_ids[j],
            "local_id": indice.local_ids[j],
        })
    return out


if __name__ == "__main__":
    import sys
    # Uso: python hash_db.py            -> costruisce TUTTO (pesante)
    #      python hash_db.py swsh3 sv03 -> costruisce solo quei set
    set_richiesti = sys.argv[1:] or None
    costruisci(sets=set_richiesti)

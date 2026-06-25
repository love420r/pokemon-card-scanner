import cv2
import os
import easyocr
import requests
import numpy as np
from rapidfuzz import fuzz
import re
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import imagehash


def _maschera_bordi(immagine):
    # Mappa dei bordi usata da tutta la pipeline di visione:
    # grigio -> sfoca -> Canny -> dilata (salda i bordi spezzati di sfondi
    # chiari e bustine lucide). Estratta qui per non duplicarla ovunque.
    grigia = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
    sfocata = cv2.GaussianBlur(grigia, (5, 5), 0)
    bordi = cv2.Canny(sfocata, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordi = cv2.dilate(bordi, kernel, iterations=1)
    return bordi


def carica_e_ridimensiona(percorso, larghezza_target):
    # Carica l'immagine dal percorso
    immagine = cv2.imread(percorso)

    # Controllo: se non si carica, restituisci None
    if immagine is None:
        print("ERRORE: immagine non trovata o non leggibile")
        return None

    # Calcola l'altezza proporzionale
    altezza_originale = immagine.shape[0]
    larghezza_originale = immagine.shape[1]
    rapporto = larghezza_target / larghezza_originale
    altezza_target = int(altezza_originale * rapporto)

    # Ridimensiona
    immagine_piccola = cv2.resize(immagine, (larghezza_target, altezza_target))

    return immagine_piccola


def trova_rettangoli_carte(immagine, area_minima):
    bordi = _maschera_bordi(immagine)
    contorni, _ = cv2.findContours(bordi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rettangoli = []
    for c in contorni:
        if cv2.contourArea(c) < area_minima:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rapporto_forma = h / w
        if 1.15 <= rapporto_forma <= 1.65:
            rettangoli.append((x, y, w, h))
        elif 2.3 <= rapporto_forma <= 3.3:
            meta = h // 2
            rettangoli.append((x, y, w, meta))
            rettangoli.append((x, y + meta, w, meta))
    return rettangoli

def trova_rettangoli_ruotati(immagine, area_minima):
    bordi = _maschera_bordi(immagine)
    contorni, _ = cv2.findContours(bordi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contorni:
        if cv2.contourArea(c) < area_minima:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if rw == 0 or rh == 0:
            continue
        lungo, corto = max(rw, rh), min(rw, rh)
        if 1.15 <= lungo / corto <= 1.65:
            rects.append(rect)
    return rects

def proponi_griglie(blob_w, blob_h, n_carte, tolleranza=1):
    # Dato un blob (larghezza, altezza) e quante carte dice l'utente,
    # propone le griglie (righe × colonne) plausibili.
    # Una cella-carta deve avere rapporto lato_lungo/lato_corto ~1.4.
    RAPPORTO_CARTA = 1.4
    proposte = []

    # Provo tutte le combinazioni righe×colonne che danno ~n_carte
    for righe in range(1, n_carte + 1):
        for colonne in range(1, n_carte + 1):
            totale = righe * colonne
            # L'utente può aver sbagliato di ±tolleranza
            if abs(totale - n_carte) > tolleranza:
                continue
            # Quanto sarebbe grande una cella con questa griglia?
            cella_w = blob_w / colonne
            cella_h = blob_h / righe
            lungo, corto = max(cella_w, cella_h), min(cella_w, cella_h)
            rapporto = lungo / corto
            # Quanto si discosta dalla forma di una carta?
            errore = abs(rapporto - RAPPORTO_CARTA)
            proposte.append((errore, righe, colonne, totale, round(rapporto, 2)))

    proposte.sort()   # la più "a forma di carta" per prima
    return proposte

def spacca_blob_in_celle(immagine_originale, rect_blob, righe, colonne, rapporto, cartella_output, start_index=0):
    # Spacca un blob fuso (rect = minAreaRect in scala ridotta) in righe×colonne celle,
    # raddrizzando il blob e ritagliando ogni cella. Salva e ritorna i ritagli.
    os.makedirs(cartella_output, exist_ok=True)

    (cx, cy), (rw, rh), angolo = rect_blob
    # porta in scala originale
    cx, cy = cx * rapporto, cy * rapporto
    rw, rh = rw * rapporto, rh * rapporto

    # normalizza l'angolo (stessa logica di ritaglia_e_raddrizza)
    if angolo < -45:
        angolo = angolo + 90
        rw, rh = rh, rw

    # ruota tutta la foto per raddrizzare il blob
    h, w = immagine_originale.shape[0], immagine_originale.shape[1]
    M = cv2.getRotationMatrix2D((cx, cy), angolo, 1.0)
    ruotata = cv2.warpAffine(immagine_originale, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # angolo in alto a sinistra del blob raddrizzato
    x0 = int(cx - rw / 2)
    y0 = int(cy - rh / 2)
    cella_w = rw / colonne
    cella_h = rh / righe

    ritagli = []
    idx = start_index
    for r in range(righe):
        for c in range(colonne):
            x = int(x0 + c * cella_w)
            y = int(y0 + r * cella_h)
            cella = ruotata[y:y + int(cella_h), x:x + int(cella_w)]
            if cella.size == 0:
                continue
            # se la cella è venuta sdraiata, raddrizza in piedi (come per le singole)
            if cella.shape[1] > cella.shape[0]:
                cella = cv2.rotate(cella, cv2.ROTATE_90_CLOCKWISE)
            nome_file = os.path.join(cartella_output, f"carta_{idx}.jpg")
            cv2.imwrite(nome_file, cella)
            ritagli.append(cella)
            idx += 1
    return ritagli

# --- Rifinitura cella d'album (carta dentro busta) --------------------------
# Il bordo della carta NON si isola col contorno più grande: in una tasca la
# carta+busta riempie quasi tutta la cella, quindi il contorno grande è la cella
# stessa e la busta (anch'essa ~1.4) sta appena fuori dalla carta. La leva che
# funziona è TAGLIARE le bande scure (solco busta/divisori del raccoglitore e la
# striscia della carta vicina che sborda) dai 4 lati. Costanti tarate sul banco
# bench_rifinitura.py (carte4.jpg).
_RIF_DARK = 95        # sotto questo grigio un pixel è "scuro" (busta/divisorio/sfondo)
_RIF_FRAC = 0.85      # frazione di pixel scuri perché una riga/colonna sia "banda da tagliare"
_RIF_SEARCH = 0.50    # entro quale quota di ciascun bordo cercare le bande scure


def _raddrizza_cella(cella):
    # Raddrizza la cella con l'angolo del minAreaRect del blob luminoso (carta+busta).
    # Se l'angolo è trascurabile o sospetto (>15°), lascia com'è: una rotazione
    # piccola introduce artefatti di interpolazione che ALZANO il pHash più di quanto
    # il raddrizzamento aiuti.
    g = cv2.cvtColor(cella, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    blur = cv2.GaussianBlur(g, (7, 7), 0)
    _, th = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    contorni, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contorni:
        return cella
    big = max(contorni, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(big)
    if rw == 0 or rh == 0:
        return cella
    if ang < -45:
        ang += 90
    if abs(ang) < 0.3 or abs(ang) > 15:
        return cella
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    return cv2.warpAffine(cella, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _taglia_bande_scure(cella):
    # Vicino a ciascun bordo cerca l'ultima/prima "banda scura" (riga/colonna con
    # ≥ _RIF_FRAC di pixel scuri): è il solco busta/divisorio o la striscia della
    # carta vicina che sborda. Ritorna (xl, yt, xr, yb) del rettangolo carta.
    H, W = cella.shape[:2]
    g = cv2.cvtColor(cella, cv2.COLOR_BGR2GRAY)
    righe_scure = (g < _RIF_DARK).mean(axis=1)     # frazione scura per riga
    colonne_scure = (g < _RIF_DARK).mean(axis=0)   # frazione scura per colonna

    lim = int(H * _RIF_SEARCH)
    b = np.where(righe_scure[:lim] > _RIF_FRAC)[0]
    yt = int(b.max()) + 1 if len(b) else 0
    b = np.where(righe_scure[H - lim:] > _RIF_FRAC)[0]
    yb = H - lim + int(b.min()) if len(b) else H

    limc = int(W * _RIF_SEARCH)
    b = np.where(colonne_scure[:limc] > _RIF_FRAC)[0]
    xl = int(b.max()) + 1 if len(b) else 0
    b = np.where(colonne_scure[W - limc:] > _RIF_FRAC)[0]
    xr = W - limc + int(b.min()) if len(b) else W

    if xr - xl < W * 0.3 or yb - yt < H * 0.3:     # trim degenere: non tagliare
        return 0, 0, W, H
    return xl, yt, xr, yb


def rifinisci_cella(cella):
    # Stringe una cella sul bordo carta per il pHash: raddrizza → taglia le bande
    # scure (busta/divisori/striscia vicina) dai 4 lati. Con le celle già allineate
    # ai bordi veri delle carte (estrai_griglia usa il profilo di proiezione), basta
    # questo: la vecchia normalizzazione di rapporto centrata sovra-processava le
    # celle ben inquadrate e ALZAVA il pHash. Su errore ritorna la cella originale.
    try:
        c = _raddrizza_cella(cella)
        xl, yt, xr, yb = _taglia_bande_scure(c)
        sub = c[yt:yb, xl:xr]
        return sub if sub.size else cella
    except Exception:
        return cella

def _bande_da_profilo(profilo, frazione_min=0.04):
    # Dato un profilo di luminosità (1 valore per colonna o per riga), trova i
    # tratti "chiari" (le carte) separati da tratti "scuri" (divisori/sfondo).
    # Ritorna la lista di (inizio, fine) delle bande chiare.
    p = np.asarray(profilo, dtype=float)
    lo, hi = np.percentile(p, 15), np.percentile(p, 85)
    soglia = (lo + hi) / 2.0

    bande = []
    in_banda = False
    inizio = 0
    for i, v in enumerate(p):
        if v >= soglia and not in_banda:
            inizio, in_banda = i, True
        elif v < soglia and in_banda:
            bande.append((inizio, i))
            in_banda = False
    if in_banda:
        bande.append((inizio, len(p)))

    # Scarta le bande troppo strette (rumore, non una carta)
    larghezza_min = len(p) * frazione_min
    return [(a, b) for (a, b) in bande if (b - a) >= larghezza_min]


def rileva_griglia(immagine):
    # Trova le bande di carte lungo le righe (y) e le colonne (x) con il
    # profilo di proiezione. Ritorna (bande_righe, bande_colonne).
    grigia = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
    profilo_colonne = grigia.mean(axis=0)   # media per ogni x  -> colonne
    profilo_righe = grigia.mean(axis=1)     # media per ogni y  -> righe
    bande_righe = _bande_da_profilo(profilo_righe)
    bande_colonne = _bande_da_profilo(profilo_colonne)
    return bande_righe, bande_colonne


def _bande_uniformi(estensione, n):
    # Divide (inizio, fine) in n tratti uguali. Ultimo ripiego se non si trovano
    # i bordi veri delle carte.
    a, b = estensione
    passo = (b - a) / n
    return [(int(a + k * passo), int(a + (k + 1) * passo)) for k in range(n)]


def _bande_da_nomi(posizioni, dim):
    # Costruisce le bande dalle posizioni dei NOMI (centri delle righe/colonne in
    # frazione 0-1). Il nome sta in cima alla carta: la banda di una riga va da
    # poco sopra il suo nome a poco sopra il nome successivo; l'ultima dura un passo.
    pos = sorted(p * dim for p in posizioni)
    diffs = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
    passo = (sum(diffs) / len(diffs)) if diffs else dim / max(1, len(pos))
    off = 0.12 * passo                      # il nome è ~12% sotto il bordo alto carta
    bande = []
    for i in range(len(pos)):
        y0 = pos[i] - off
        y1 = (pos[i + 1] - off) if i + 1 < len(pos) else (pos[i] - off + passo)
        bande.append((max(0, int(y0)), min(dim, int(y1))))
    return bande


def _estendi_bande_corte(bande, dim):
    # Una carta TAGLIATA dal bordo tasca produce una banda molto più corta delle
    # altre (es. l'ultima riga). La estendo all'altezza mediana (la carta sfora
    # sotto il punto dove il profilo si è spento), senza superare l'immagine.
    if len(bande) < 2:
        return bande
    lunghezze = [b - a for a, b in bande]
    med = sorted(lunghezze)[len(lunghezze) // 2]
    out = []
    for a, b in bande:
        if (b - a) < 0.7 * med:
            b = min(int(a + med), dim)
        out.append((a, b))
    return out


def _concilia_bande(bande, n_atteso, posizioni, dim):
    # Concilia le bande del profilo colori col numero di righe/colonne dedotto dai
    # NOMI. Se ne trova troppe (un divisore ha spezzato una carta) fonde i gap più
    # piccoli finché tornano n_atteso; se troppo poche (due carte fuse) si fida dei
    # nomi e divide sulle loro posizioni; se non può, divisione uniforme. Infine
    # estende le bande corte (carte tagliate dal bordo tasca).
    if not n_atteso:
        return _estendi_bande_corte(bande, dim)
    bande = list(bande)
    while len(bande) > n_atteso:
        gaps = [(bande[i + 1][0] - bande[i][1], i) for i in range(len(bande) - 1)]
        _, idx = min(gaps)
        bande[idx] = (bande[idx][0], bande[idx + 1][1])   # fonde idx con idx+1
        del bande[idx + 1]
    if len(bande) == n_atteso:
        return _estendi_bande_corte(bande, dim)
    if posizioni and len(posizioni) == n_atteso:
        return _bande_da_nomi(posizioni, dim)             # mi fido del numero dei nomi
    est = (bande[0][0], bande[-1][1]) if bande else (0, dim)
    return _bande_uniformi(est, n_atteso)


def estrai_griglia(immagine_ridotta, originale, rapporto, cartella_output,
                   righe_forzate=None, colonne_forzate=None,
                   righe_y=None, colonne_x=None, rifinisci=True):
    # Estrae le carte da una PAGINA DI RACCOGLITORE seguendo i BORDI VERI delle
    # carte (profilo di proiezione di rileva_griglia), NON una divisione uniforme:
    # le tasche dell'album non sono a intervalli uguali, quindi il taglio uniforme
    # storce (taglia nome in cima e codice in fondo). righe_forzate/colonne_forzate =
    # numero di righe/colonne atteso (dai nomi o dall'override --griglia);
    # righe_y/colonne_x = posizioni dei nomi (frazione 0-1), usate come àncora se il
    # profilo non concorda col numero atteso.
    os.makedirs(cartella_output, exist_ok=True)
    bande_righe, bande_colonne = rileva_griglia(immagine_ridotta)

    if not bande_righe or not bande_colonne:
        return []   # non sembra una griglia: lascia decidere al chiamante

    H, W = immagine_ridotta.shape[0], immagine_ridotta.shape[1]
    bande_righe = _concilia_bande(bande_righe, righe_forzate, righe_y, H)
    bande_colonne = _concilia_bande(bande_colonne, colonne_forzate, colonne_x, W)

    print(f"  bande righe usate (scala ridotta):   {bande_righe}")
    print(f"  bande colonne usate (scala ridotta): {bande_colonne}")

    carte = []
    idx = 0
    for (y0, y1) in bande_righe:
        for (x0, x1) in bande_colonne:
            X0, Y0 = int(x0 * rapporto), int(y0 * rapporto)
            X1, Y1 = int(x1 * rapporto), int(y1 * rapporto)
            carta = originale[Y0:Y1, X0:X1]
            if carta.size == 0:
                continue
            if rifinisci:
                carta = rifinisci_cella(carta)
            cv2.imwrite(os.path.join(cartella_output, f"carta_{idx}.jpg"), carta)
            carte.append(carta)
            idx += 1
    return carte


# Parole comuni di testo-regole/attacchi (IT/EN): NON sono nomi di Pokémon.
# Servono a sgrossare il rumore dell'OCR prima ancora di interrogare il DB.
DIZIONARIO_RUMORE = {
    "cerca", "mazzo", "aiuto", "base", "uno", "una", "che", "puoi", "giocare",
    "carta", "carte", "turno", "durante", "assegnale", "strumento", "pokemon",
    "pokémon", "allenatore", "tuo", "del", "nel", "fino", "mostrali", "oggetto",
    "questo", "dei", "tre", "scelgo", "energia", "danni", "sola", "volta", "tuoi",
    "thunder", "powerful", "spark", "active", "this", "does", "attacco", "rapido",
    "abilità", "resistenza", "debolezza", "ritirata", "evolve", "from",
}


def _pulisci_nome_ocr(testo):
    # Normalizza un testo OCR prima di valutarlo come nome di carta.
    t = testo.strip().strip("'").strip()
    # L'OCR scambia il simbolo EX della carta con "€": "Articuno €" -> "Articuno e".
    t = t.replace("€", "e").replace("£", "e")
    # A volte attacca il codice set dopo uno slash: tienine solo la parte nome.
    if "/" in t:
        t = t.split("/")[0].strip()
    return t


def trova_nomi_e_posizioni(immagine, reader, soglia_conf=0.55):
    # OCR su TUTTA l'immagine + filtro LOCALE (veloce, senza rete): sgrossa il
    # rumore e ritorna i candidati-nome con la loro posizione (x,y) in frazione 0-1.
    h, w = immagine.shape[0], immagine.shape[1]
    risultati = reader.readtext(immagine)

    candidati = []
    for box, testo, conf in risultati:
        t = _pulisci_nome_ocr(testo)
        if conf < soglia_conf or len(t) < 4:
            continue
        if t.isdigit() or t.isupper():          # numeri puri, MAIUSCOLO (ALLENATORE)
            continue
        if any(ch.isdigit() for ch in t):       # "Ps170", "20x" -> non è un nome
            continue
        if t.lower() in DIZIONARIO_RUMORE:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        candidati.append((t, sum(xs) / 4 / w, sum(ys) / 4 / h))
    return candidati


def filtra_nomi_con_db(candidati, soglia_match=70, lingua="it", workers=8, cache=None):
    # Il "giudice" vero: tiene solo i candidati che il DB TCGdex conferma come
    # carte E il cui nome letto somiglia abbastanza a quello ufficiale.
    # IMPORTANTE: fuzz.ratio (NON partial_ratio): partial_ratio darebbe 100 a
    # "Tuono" dentro "Monte del Tuono" e farebbe passare i falsi positivi.
    # Le interrogazioni API girano in PARALLELO (da ~30s a pochi secondi).
    if cache is None:
        cache = {}

    def interroga(testo):
        if testo not in cache:
            cache[testo] = cerca_carta_api(testo, lingua)
        return cache[testo]

    nomi_pos = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futuri = {ex.submit(interroga, t): (t, cx, cy) for (t, cx, cy) in candidati}
        for fut, (t, cx, cy) in futuri.items():
            trovati = fut.result()
            if not trovati:
                continue
            nome_ufficiale = trovati[0].get("name", "")
            if fuzz.ratio(t.lower(), nome_ufficiale.lower()) >= soglia_match:
                nomi_pos.append((t, cx, cy))
    return nomi_pos


def conta_carte_ben_formate(immagine, area_minima):
    # Quante carte SINGOLE ben formate (rapporto ~1.4) vede la visione a contorni.
    # È il segnale per decidere il ramo: tante carte ben formate = "staccate"
    # (la v1 funziona); poche/zero = "attaccate/album" (serve la griglia dai nomi).
    bordi = _maschera_bordi(immagine)
    contorni, _ = cv2.findContours(bordi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in contorni:
        if cv2.contourArea(c) < area_minima:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw == 0 or rh == 0:
            continue
        if 1.15 <= max(rw, rh) / min(rw, rh) <= 1.65:
            n += 1
    return n


def assegna_nomi_a_celle(nomi_pos, righe_y, colonne_x):
    # In modalità album i nomi li abbiamo già letti: invece di ri-OCR-are ogni
    # cella, assegno ogni nome alla cella più vicina (riga, colonna). Ritorna una
    # lista lunga righe×colonne (ordine row-major, come estrai_griglia), ogni
    # elemento è la lista di nomi caduti in quella cella ([] se cella "vuota").
    righe, colonne = len(righe_y), len(colonne_x)
    celle = [[] for _ in range(righe * colonne)]
    if righe == 0 or colonne == 0:
        return celle
    for (testo, cx, cy) in nomi_pos:
        r = min(range(righe), key=lambda i: abs(cy - righe_y[i]))
        c = min(range(colonne), key=lambda i: abs(cx - colonne_x[i]))
        celle[r * colonne + c].append(testo)
    return celle


def deduci_griglia(nomi_pos, foto_w, foto_h, delta_y=0.08, delta_x=0.12, rapporto_carta=1.4):
    # nomi_pos = lista di (testo, cx, cy) in frazione 0-1
    # Ritorna (righe, colonne, righe_y, colonne_x)
    if not nomi_pos:
        return 0, 0, [], []

    # --- RIGHE: raggruppo per y vicine ---
    ys = sorted(p[2] for p in nomi_pos)
    righe_y = [ys[0]]
    for y in ys[1:]:
        if y - righe_y[-1] > delta_y:
            righe_y.append(y)
    n_righe = len(righe_y)

    # --- COLONNE: raggruppo per x vicine (come le righe, ma sull'asse x) ---
    xs = sorted(p[1] for p in nomi_pos)
    colonne_x = [xs[0]]
    for x in xs[1:]:
        if x - colonne_x[-1] > delta_x:
            colonne_x.append(x)
    n_colonne = len(colonne_x)

    return n_righe, n_colonne, righe_y, colonne_x

def ritaglia_e_salva(immagine, contorni, cartella_output):
    # Crea la cartella se non esiste
    os.makedirs(cartella_output, exist_ok=True)

    # Lista dove accumuliamo i ritagli (in memoria)
    ritagli = []

    # Per ogni contorno
    for i, (x, y, w, h) in enumerate(contorni):

        # Ritaglia
        carta = immagine[y:y+h, x:x+w]

        # Costruisci il nome file (es: "ritagli/carta_0.jpg")
        nome_file = os.path.join(cartella_output, f"carta_{i}.jpg")

        # Salva su disco
        cv2.imwrite(nome_file, carta)

        # Aggiungi alla lista in memoria
        ritagli.append(carta)

    return ritagli

def ritaglia_e_salva_fullres(immagine_originale, contorni, rapporto, cartella_output):
    # Come ritaglia_e_salva, MA ritaglia dalla foto originale full-res.
    # rapporto = quanto è più grande l'originale rispetto alla ridotta
    os.makedirs(cartella_output, exist_ok=True)
    ritagli = []
    for i, (x, y, w, h) in enumerate(contorni):
        # Riporta le coordinate alla scala dell'originale
        x = int(x * rapporto)
        y = int(y * rapporto)
        w = int(w * rapporto)
        h = int(h * rapporto)
        carta = immagine_originale[y:y+h, x:x+w]
        nome_file = os.path.join(cartella_output, f"carta_{i}.jpg")
        cv2.imwrite(nome_file, carta)
        ritagli.append(carta)
    return ritagli

def ritaglia_e_raddrizza(immagine_originale, contorni_ruotati, rapporto, cartella_output):
    # Ritaglia OGNI carta dalla foto full-res E la raddrizza usando il suo angolo.
    # contorni_ruotati = lista di minAreaRect (uno per carta), in scala ridotta
    os.makedirs(cartella_output, exist_ok=True)
    ritagli = []

    for i, rect in enumerate(contorni_ruotati):
        (cx, cy), (rw, rh), angolo = rect

        # Riporta centro e dimensioni alla scala dell'originale full-res
        cx, cy = cx * rapporto, cy * rapporto
        rw, rh = rw * rapporto, rh * rapporto

        # Normalizza l'angolo strano di OpenCV
        if angolo < -45:
            angolo = angolo + 90
            rw, rh = rh, rw          # scambia le dimensioni insieme all'angolo

        # Ruota tutta la foto attorno al centro della carta per raddrizzarla
        h, w = immagine_originale.shape[0], immagine_originale.shape[1]
        M = cv2.getRotationMatrix2D((cx, cy), angolo, 1.0)
        ruotata = cv2.warpAffine(immagine_originale, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Ora la carta è dritta: ritaglia col rettangolo (centro + dimensioni)
        rw, rh = int(rw), int(rh)
        x = int(cx - rw / 2)
        y = int(cy - rh / 2)
        carta = ruotata[y:y+rh, x:x+rw]

        if carta.size == 0:
            continue

        # Il TUO controllo: se è venuta sdraiata, raddrizza in piedi
        if carta.shape[1] > carta.shape[0]:
            carta = cv2.rotate(carta, cv2.ROTATE_90_CLOCKWISE)

        nome_file = os.path.join(cartella_output, f"carta_{i}.jpg")
        cv2.imwrite(nome_file, carta)
        ritagli.append(carta)

    return ritagli

def raddrizza_carta(immagine):
    # Raddrizza una carta storta. Ritorna l'immagine ruotata dritta.
    grigia = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
    _, soglia = cv2.threshold(grigia, 1, 255, cv2.THRESH_BINARY)
    punti = cv2.findNonZero(soglia)
    if punti is None:
        return immagine

    # Rettangolo RUOTATO minimo che abbraccia la carta: dà centro, (w,h), angolo
    rect = cv2.minAreaRect(punti)
    (cx, cy), (rw, rh), angolo = rect

    # Normalizza l'angolo grezzo di OpenCV in un piccolo aggiustamento
    if angolo < -45:
        angolo = angolo + 90
        print(f"  [debug] angolo applicato: {angolo:.1f}")

    # Ruota l'immagine attorno al suo centro per annullare l'inclinazione
    h, w = immagine.shape[0], immagine.shape[1]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angolo, 1.0)
    dritta = cv2.warpAffine(immagine, M, (w, h),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # IL TUO CONTROLLO: se è venuta sdraiata (più larga che alta), girala di 90°
    dh, dw = dritta.shape[0], dritta.shape[1]
    if dw > dh:
        dritta = cv2.rotate(dritta, cv2.ROTATE_90_CLOCKWISE)

    return dritta


SOGLIA_FUZZY = 75   # quanto deve somigliare il nome (0-100) per tenere il candidato

def estrai_codice_set(testi):
    # testi è la lista di tuple (testo, confidenza) dall'OCR
    # cerca un codice tipo "020/094"; ritorna la stringa o None se non c'è
    pattern = r"\d+/\d+"
    for testo, confidenza in testi:
        trovato = re.search(pattern, testo)
        if trovato:
            return trovato.group()
    return None

def leggi_codice_set(percorso, reader):
    # Legge il codice "73/116" dalla striscia bassa. Niente ROI stretta:
    # tutta la striscia + regex che trova il pattern ovunque. Ritorna stringa o None.
    immagine = cv2.imread(percorso)
    if immagine is None:
        return None

    h = immagine.shape[0]
    striscia = immagine[int(h * 0.82):, :]          # tutta la larghezza, ultimo 18%

    grigia = cv2.cvtColor(striscia, cv2.COLOR_BGR2GRAY)
    grande = cv2.resize(grigia, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    risultati = reader.readtext(grande, allowlist="0123456789/")

    # Unisco tutti i pezzi letti in una stringa sola, poi cerco il pattern NNN/NNN
    testo_unito = " ".join(t for _, t, _ in risultati)
    testo_unito = testo_unito.replace(" ", "")
    trovato = re.search(r"(\d{1,3})/(\d{2,3})", testo_unito)
    if trovato:
        return f"{trovato.group(1)}/{trovato.group(2)}"
    return None

def leggi_nome_carta(percorso, reader, frazione=0.15):
    # OCR mirato sulla fascia ALTA della carta, dove vive il nome.
    # Ritorna lista di stringhe candidate (ordinate per confidenza), o [].
    immagine = cv2.imread(percorso)
    if immagine is None:
        return []

    h = immagine.shape[0]
    fascia = immagine[:int(h * frazione), :]        # primo 15% dall'alto, tutta la larghezza

    grigia = cv2.cvtColor(fascia, cv2.COLOR_BGR2GRAY)
    grande = cv2.resize(grigia, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    risultati = reader.readtext(grande)

    # Ordina per confidenza (più sicuro prima) e tieni solo testo "da nome"
    risultati.sort(key=lambda x: -x[2])
    candidati = []
    for _, testo, conf in risultati:
        pulito = testo.strip("'").strip()
        if "/" in pulito:                    # taglia la coda dopo lo slash
            pulito = pulito.split("/")[0].strip()
        if len(pulito) < 3:
            continue
        if pulito.isdigit():
            continue
        candidati.append(pulito)
    return candidati

def _chiama_tcgdex(termine, lingua):
    # Helper: fa una singola chiamata a TCGdex e ritorna la lista (o [] se errore)
    url = f"https://api.tcgdex.net/v2/{lingua}/cards"
    risposta = requests.get(url, params={"name": termine})
    if risposta.status_code != 200:
        print(f"  ⚠️ Errore API: {risposta.status_code}")
        return []
    return risposta.json()


def cerca_carta_api(nome, lingua="it"):
    nome_pulito = nome.strip("'").strip()

    # Tentativo 1: ricerca esatta
    risultati = _chiama_tcgdex(nome_pulito, lingua)
    if risultati:
        return risultati

    # Tentativo 2: zero risultati → riprova col prefisso (regola 2/3)
    lunghezza_prefisso = int(len(nome_pulito) * 2 / 3)
    prefisso = nome_pulito[:lunghezza_prefisso]
    candidati = _chiama_tcgdex(prefisso, lingua)

    # Filtra: tieni solo i candidati il cui nome somiglia abbastanza all'originale
    buoni = []
    for c in candidati:
        punteggio = fuzz.ratio(nome_pulito.lower(), c["name"].lower())
        if punteggio >= SOGLIA_FUZZY:
            buoni.append(c)
    if buoni:
        return buoni

    # Tentativo 3: l'errore è spesso in TESTA (es. "Hvenusaur"→"venusaur",
    # la "M" di Mega). Togli il primo carattere e riprova col prefisso + fuzzy.
    senza_testa = nome_pulito[1:]
    if len(senza_testa) >= 3:
        prefisso2 = senza_testa[:int(len(senza_testa) * 2 / 3)]
        candidati2 = _chiama_tcgdex(prefisso2, lingua)
        for c in candidati2:
            # confronto fuzzy sul nome SENZA testa, perdona l'errore iniziale
            punteggio = fuzz.ratio(senza_testa.lower(), c["name"].lower())
            if punteggio >= SOGLIA_FUZZY:
                buoni.append(c)

    return buoni

def scarica_sets(lingua="it"):
    try:
        risposta = requests.get(f"https://api.tcgdex.net/v2/{lingua}/sets", timeout=10)
        return risposta.json() if risposta.status_code == 200 else []
    except requests.exceptions.RequestException:
        print("  ⚠️ Rete: impossibile scaricare i set")
        return []


def cerca_per_codice(numero_carta, totale_set, lista_sets, lingua="it", tolleranza=2):
    candidati = []
    for s in lista_sets:
        official = s.get("cardCount", {}).get("official")
        if official is None:
            continue
        if abs(official - totale_set) <= tolleranza:
            id_carta = f"{s['id']}-{int(numero_carta):03d}"
            try:
                r = requests.get(f"https://api.tcgdex.net/v2/{lingua}/cards/{id_carta}", timeout=10)
                if r.status_code == 200:
                    candidati.append(r.json())
            except requests.exceptions.RequestException:
                continue
    return candidati

SOGLIA_MAX = 18      # oltre questa distanza pHash, il match non è affidabile
STACCO_MIN = 6       # il 1° deve staccare il 2° di almeno tanto

# --- ORB: giudice fine sui casi duri (olografiche, dove il pHash ha un floor) ---
# Misurato sul banco bench_orb.py: quando il pHash resta > 13, il feature matching
# ORB (inlier geometricamente coerenti) separa il candidato giusto. Soglie tarate:
# Articuno 278, Drampa 229, Martes 370 confermano; Pikachu-sbagliato (83 vs 79,
# stacco 4) e i foil distrutti (16, 6) restano sotto soglia+stacco -> Uncertain.
SOGLIA_ORB = 70      # inlier minimi per fidarsi di ORB
STACCO_ORB = 25      # il 1° candidato ORB deve staccare il 2° di almeno tanto
ORB_TOPK = 30        # quanti candidati (i più vicini per pHash) passare a ORB
ORB_CACHE = "orb_cache"   # cache su disco delle immagini full-res dei riferimenti


def _orb_grigia(img_bgr, alt=700):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if h != alt and h > 0:
        g = cv2.resize(g, (max(1, int(w * alt / h)), alt), interpolation=cv2.INTER_AREA)
    return g


def _orb_inliers(g1, g2, nfeat=2000):
    # ORB + ratio test + RANSAC: ritorna il numero di inlier (match coerenti).
    orb = cv2.ORB_create(nfeat)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for mn in bf.knnMatch(d1, d2, k=2):
        if len(mn) == 2 and mn[0].distance < 0.75 * mn[1].distance:
            good.append(mn[0])
    if len(good) < 4:
        return len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def _scarica_ref_orb(url_base, card_id):
    # Scarica (e mette in cache) l'immagine full-res /high.png del riferimento.
    os.makedirs(ORB_CACHE, exist_ok=True)
    path = os.path.join(ORB_CACHE, f"{card_id}.png")
    if os.path.exists(path):
        return cv2.imread(path)
    for suff in ("/high.png", "/low.png"):
        try:
            r = requests.get(url_base + suff, timeout=20)
        except requests.exceptions.RequestException:
            continue
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return cv2.imread(path)
    return None


def _orb_rerank(percorso_locale, risultati, img_per_id):
    # Ri-ordina i candidati (i più vicini per pHash) per inlier ORB decrescenti.
    # Ritorna lista di (inlier, risultato_tupla). risultato = (dist, nome, id, localId).
    cell = cv2.imread(percorso_locale)
    if cell is None:
        return []
    g_cell = _orb_grigia(cell)
    out = []
    for r in risultati[:ORB_TOPK]:
        url = img_per_id.get(r[2])
        if not url:
            continue
        ref = _scarica_ref_orb(url, r[2])
        if ref is None:
            continue
        out.append((_orb_inliers(g_cell, _orb_grigia(ref)), r))
    out.sort(key=lambda x: -x[0])
    return out


def identifica_carta(percorso_locale, nome_ocr, reader=None, lista_sets=None, usa_orb=False):
    img_locale = Image.open(percorso_locale)
    hash_locale = imagehash.phash(img_locale)

    # nome_ocr può essere stringa singola o lista di tentativi
    if isinstance(nome_ocr, str):
        nomi_da_provare = [nome_ocr]
    else:
        nomi_da_provare = nome_ocr

    # Terzo strato: leggi il codice una volta sola (non dipende dal nome)
    codice = None
    if reader is not None:
        codice = leggi_codice_set(percorso_locale, reader)


    miglior_esito = {"stato": "FALLITO", "motivo": "nessun candidato", "carta": None}

    for tentativo in nomi_da_provare:
        candidati = cerca_carta_api(tentativo)
        if not candidati:
            continue   # questo nome non porta a niente, prova il prossimo

        # Classifica pHash dei candidati di QUESTO nome
        risultati = []
        for c in candidati:
            url_base = c.get("image")
            if url_base is None:
                continue
            risposta = requests.get(url_base + "/low.png")
            if risposta.status_code != 200:
                continue
            hash_c = imagehash.phash(Image.open(BytesIO(risposta.content)))
            risultati.append((hash_locale - hash_c, c["name"], c["id"], c.get("localId", "")))

        if not risultati:
            continue

        risultati.sort()

        # Il giudice. Tra i candidati del nome cerca quello col localId == numero
        # letto. Se il pHash è affidabile basta quello; se è ALTO (olografiche: il
        # pHash ha un floor) la conferma vale lo stesso PURCHÉ combaci anche il
        # TOTALE del set — perché nome+numero+totale identificano la carta da soli.
        esito = None
        if codice is not None and "/" in codice:
            numero, totale = codice.split("/")
            tot_per_set = {}
            if lista_sets:
                for s in lista_sets:
                    off = s.get("cardCount", {}).get("official")
                    if off is not None:
                        tot_per_set[s["id"]] = off
            conferme = []
            for r in risultati:
                local_id = r[3]
                if not (local_id.isdigit() and int(local_id) == int(numero)):
                    continue
                set_id = r[2].rsplit("-", 1)[0]
                totale_ok = abs(tot_per_set.get(set_id, -999) - int(totale)) <= 2
                # conferma se il pHash è affidabile OPPURE il totale del set combacia
                if r[0] <= SOGLIA_MAX or totale_ok:
                    conferme.append((not totale_ok, r))   # totale combaciante per primo
            if conferme:
                conferme.sort(key=lambda x: (x[0], x[1][0]))   # totale-ok, poi pHash minore
                migliore = conferme[0][1]
                nota = "codice+totale" if not conferme[0][0] else "codice"
                esito = {"stato": "OK",
                         "motivo": f"{nota} {codice} conferma (pHash {migliore[0]})",
                         "carta": migliore}

        # Logica pHash pura (se il giudice si è astenuto)
        if esito is None:
            migliore = risultati[0]
            distanza_top = migliore[0]
            if distanza_top <= 13:
                esito = {"stato": "OK", "motivo": f"distanza {distanza_top}", "carta": migliore}
            elif distanza_top <= SOGLIA_MAX:
                esito = {"stato": "INCERTO", "motivo": f"distanza {distanza_top} (incerta)", "carta": migliore}
            else:
                esito = {"stato": "INCERTO", "motivo": f"distanza {distanza_top} troppo alta", "carta": migliore}

        # GIUDICE ORB: solo casi duri (pHash NON ha dato OK, distanza > 13) e solo se
        # richiesto (album). Il feature matching separa le olografiche dove il pHash
        # ha un floor. Conferma OK solo con inlier >= soglia E stacco sul 2°; altrimenti
        # resta Uncertain (niente conferme inventate).
        if usa_orb and esito["stato"] != "OK":
            img_per_id = {c["id"]: c.get("image") for c in candidati}
            orb_ris = _orb_rerank(percorso_locale, risultati, img_per_id)
            if orb_ris:
                top1_inl, top1_r = orb_ris[0]
                top2_inl = orb_ris[1][0] if len(orb_ris) > 1 else 0
                if top1_inl >= SOGLIA_ORB and (top1_inl - top2_inl) >= STACCO_ORB:
                    esito = {"stato": "OK",
                             "motivo": f"ORB {top1_inl} inlier (stacco {top1_inl - top2_inl}, pHash {top1_r[0]})",
                             "carta": top1_r,
                             # confidenza ORB: il report la usa per l'affidabilità%
                             # (il pHash resta alto sulle holo e darebbe una % fuorviante)
                             "orb": (top1_inl, top1_inl - top2_inl)}

        # CASCATA: OK → abbiamo finito. Altrimenti ricorda e prova il prossimo nome
        if esito["stato"] == "OK":
            return esito
        if miglior_esito["carta"] is None:
            miglior_esito = esito   # tieni il primo esito concreto come ripiego

    # SECONDA GAMBA: nessun nome ha dato OK → prova per CODICE
    if miglior_esito["stato"] != "OK" and codice is not None and lista_sets:
        numero, totale = codice.split("/")
        candidati = cerca_per_codice(int(numero), int(totale), lista_sets)

        risultati = []
        for c in candidati:
            url_base = c.get("image")
            if url_base is None:
                continue
            try:
                risposta = requests.get(url_base + "/low.png", timeout=10)
            except requests.exceptions.RequestException:
                continue
            if risposta.status_code != 200:
                continue
            hash_c = imagehash.phash(Image.open(BytesIO(risposta.content)))
            risultati.append((hash_locale - hash_c, c["name"], c["id"], c.get("localId", "")))

        if risultati:
            risultati.sort()
            migliore = risultati[0]
            if migliore[0] <= SOGLIA_MAX:
                return {"stato": "OK",
                        "motivo": f"code-first: {codice} (pHash {migliore[0]})",
                        "carta": migliore}
            else:
                miglior_esito = {"stato": "INCERTO",
                                 "motivo": f"code-first distanza {migliore[0]}",
                                 "carta": migliore}

    return miglior_esito


# --- Identificazione tramite DB pHash LOCALE (veloce, offline) ---------------
SOGLIA_LOCALE_OK = 12        # pHash entro cui il match è affidabile
SOGLIA_LOCALE_INCERTO = 20   # oltre questo, non ci fidiamo
STACCO_LOCALE = 5            # il 1° deve staccare il 2° di almeno tanto


def identifica_carta_locale(percorso_locale, indice, reader=None, top_n=6):
    # Identifica una carta confrontando il suo pHash con l'indice LOCALE
    # (vedi hash_db.py). L'OCR del nome serve solo come spareggio se il
    # pHash è vicino ma ambiguo. Ritorna lo stesso formato di identifica_carta:
    #   {"stato", "motivo", "carta": (distanza, nome, id, local_id)}
    import hash_db

    ph = imagehash.phash(Image.open(percorso_locale))
    risultati = hash_db.cerca(ph, indice, top_n=top_n)
    if not risultati:
        return {"stato": "FALLITO", "motivo": "indice vuoto / nessun match", "carta": None}

    def impacchetta(r, stato, motivo):
        carta = (r["distanza"], r["nome"], r["id"], r["local_id"])
        return {"stato": stato, "motivo": motivo, "carta": carta}

    top = risultati[0]
    d0 = top["distanza"]
    secondo = risultati[1]["distanza"] if len(risultati) > 1 else 999

    # Caso netto: vicino E ben staccato dal secondo candidato
    if d0 <= SOGLIA_LOCALE_OK and (secondo - d0) >= STACCO_LOCALE:
        return impacchetta(top, "OK", f"pHash {d0} (stacco {secondo - d0})")

    # Spareggio con OCR del nome quando il pHash è vicino ma ambiguo
    if reader is not None and d0 <= SOGLIA_LOCALE_INCERTO:
        nomi_ocr = leggi_nome_carta(percorso_locale, reader)
        for r in risultati:
            if r["distanza"] > SOGLIA_LOCALE_INCERTO:
                continue
            for n in nomi_ocr:
                if fuzz.ratio(n.lower(), r["nome"].lower()) >= 80:
                    return impacchetta(r, "OK", f"pHash {r['distanza']} + nome OCR '{n}'")

    # Nessuna conferma forte: ripiega sul più vicino, segnando l'incertezza
    if d0 <= SOGLIA_LOCALE_OK:
        return impacchetta(top, "OK", f"pHash {d0}")
    if d0 <= SOGLIA_LOCALE_INCERTO:
        return impacchetta(top, "INCERTO", f"pHash {d0} (incerto)")
    return impacchetta(top, "INCERTO", f"pHash {d0} troppo alto")


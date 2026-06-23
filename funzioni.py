import cv2
import os
import easyocr
import requests
from rapidfuzz import fuzz
import re
from PIL import Image
from io import BytesIO
import imagehash


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
    grigia = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
    sfocata = cv2.GaussianBlur(grigia, (5, 5), 0)
    bordi = cv2.Canny(sfocata, 50, 150)

    # Salda i bordi spezzati (sfondi chiari, buste lucide)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordi = cv2.dilate(bordi, kernel, iterations=1)

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
    grigia = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
    sfocata = cv2.GaussianBlur(grigia, (5, 5), 0)
    bordi = cv2.Canny(sfocata, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordi = cv2.dilate(bordi, kernel, iterations=1)
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


def identifica_carta(percorso_locale, nome_ocr, reader=None, lista_sets=None):
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

        # Il giudice (con museruola)
        esito = None
        if codice is not None:
            numero = codice.split("/")[0]
            conferme = []
            for r in risultati:
                local_id = r[3]
                if local_id.isdigit() and int(local_id) == int(numero) and r[0] <= SOGLIA_MAX:
                    conferme.append(r)
            if len(conferme) >= 1:
                migliore = conferme[0]
                esito = {"stato": "OK",
                         "motivo": f"codice {codice} conferma (pHash {migliore[0]})",
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


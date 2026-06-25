import cv2
import sys
import easyocr
import hash_db
import prezzi
from funzioni import (carica_e_ridimensiona, conta_carte_ben_formate,
                      trova_nomi_e_posizioni, filtra_nomi_con_db, deduci_griglia,
                      assegna_nomi_a_celle, estrai_griglia, trova_rettangoli_ruotati,
                      ritaglia_e_raddrizza, leggi_nome_carta, identifica_carta,
                      identifica_carta_locale, scarica_sets)

# La console di Windows usa cp1252 e va in errore sugli emoji del report.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ---------------------------------------------------------------------------
# USO (zero setup: clona, lancia, funziona — identificazione ONLINE via TCGdex):
#   python main.py foto.jpg              carte staccate o attaccate/album (auto)
#   python main.py pagina.jpg --griglia 3x3   override manuale della griglia
#   python main.py foto.jpg --offline    avanzato: usa il DB pHash locale (più
#                                        veloce, ma va costruito prima con
#                                        costruisci_db.py). Vedi README.
# ---------------------------------------------------------------------------
argv = sys.argv[1:]

offline = False
for flag in ("--offline", "--db-locale"):
    if flag in argv:
        offline = True
        argv.remove(flag)

griglia_forzata = None
if "--griglia" in argv:
    pos = argv.index("--griglia")
    try:
        r, c = argv[pos + 1].lower().split("x")
        griglia_forzata = (int(r), int(c))
    except (IndexError, ValueError):
        print("Formato --griglia non valido. Esempio: --griglia 3x3")
        sys.exit(1)
    del argv[pos:pos + 2]

if argv:
    PERCORSO_IMMAGINE = argv[0]
else:
    PERCORSO_IMMAGINE = "carte3.jpg"
    print("Nessun file specificato, uso carte3.jpg (default)\n")

LARGHEZZA_RESIZE = 1000
AREA_MINIMA = 5000
CARTELLA_RITAGLI = "ritagli"

# --- Carica l'immagine ---
originale = cv2.imread(PERCORSO_IMMAGINE)
immagine = carica_e_ridimensiona(PERCORSO_IMMAGINE, LARGHEZZA_RESIZE)
if immagine is None or originale is None:
    print("Impossibile caricare l'immagine. Esco.")
    sys.exit(1)
rapporto = originale.shape[1] / immagine.shape[1]

# --- Carica EasyOCR una volta (serve sia a segmentare l'album sia a identificare) ---
print("Caricamento modelli EasyOCR...")
reader = easyocr.Reader(['it', 'en'])
print("Pronto!\n")


def segmenta_v1():
    # Pipeline v1 (carte staccate): contorni ruotati -> ritaglio + raddrizzamento.
    rects = trova_rettangoli_ruotati(immagine, AREA_MINIMA)
    return ritaglia_e_raddrizza(originale, rects, rapporto, CARTELLA_RITAGLI)


# nomi_celle[i] = nomi già noti per la cella i (solo modalità album); None altrove
nomi_celle = None

# --- FASE 1: dalla foto ai ritagli (sceglie da solo staccate vs album) ---
if griglia_forzata:
    righe, colonne = griglia_forzata
    print(f"Griglia forzata: {righe}x{colonne}")
    ritagli = estrai_griglia(immagine, originale, rapporto, CARTELLA_RITAGLI,
                             righe_forzate=righe, colonne_forzate=colonne)
    if not ritagli:
        print("  Griglia non ritagliabile, ripiego sulla visione a contorni.")
        ritagli = segmenta_v1()
else:
    n_viste = conta_carte_ben_formate(immagine, AREA_MINIMA)
    if n_viste >= 2:
        # CARTE STACCATE: la visione v1 funziona, usala (non la tocchiamo)
        print(f"Carte staccate riconosciute dalla visione: {n_viste}")
        ritagli = segmenta_v1()
    else:
        # ATTACCATE / PAGINA D'ALBUM: deduco la griglia DAI NOMI
        print("Visione insufficiente (carte attaccate?) → deduco la griglia dai nomi...")
        candidati = trova_nomi_e_posizioni(immagine, reader)
        nomi_pos = filtra_nomi_con_db(candidati)
        righe, colonne, righe_y, colonne_x = deduci_griglia(
            nomi_pos, originale.shape[1], originale.shape[0])
        print(f"  Nomi validi trovati: {len(nomi_pos)} → griglia {righe}x{colonne}")
        ritagli = []
        if righe * colonne >= 2:
            ritagli = estrai_griglia(immagine, originale, rapporto, CARTELLA_RITAGLI,
                                     righe_forzate=righe, colonne_forzate=colonne,
                                     righe_y=righe_y, colonne_x=colonne_x)
        if ritagli:
            # I nomi li abbiamo già: assegnali alle celle (niente ri-OCR a vuoto)
            nomi_celle = assegna_nomi_a_celle(nomi_pos, righe_y, colonne_x)
        else:
            print("  Griglia non deducibile, ripiego sulla visione a contorni.")
            ritagli = segmenta_v1()

print(f"\nCarte trovate e ritagliate: {len(ritagli)}\n")

# --- FASE 2: prepara l'identificazione (online di default, offline su richiesta) ---
indice = None
lista_sets = None
if offline:
    print("Modalità OFFLINE: uso il DB pHash locale.")
    indice = hash_db.carica_indice()
    if len(indice) == 0:
        print("\n⚠️  Il DB locale è VUOTO. Costruiscilo prima, es.:")
        print("      python costruisci_db.py --set swsh3 sv03")
        print("   Oppure togli --offline per usare l'identificazione ONLINE. Esco.")
        sys.exit(1)
    print(f"Carte nell'indice: {len(indice)}\n")
else:
    print("Modalità ONLINE: identifico via TCGdex (nessun setup richiesto).")
    lista_sets = scarica_sets()
    print(f"Set scaricati per il code-first: {len(lista_sets)}\n")

cache_prezzi = prezzi.carica_cache()

# --- FASE 3: identifica ogni carta e cerca il prezzo ---
report = []
for i in range(len(ritagli)):
    percorso = f"{CARTELLA_RITAGLI}/carta_{i}.jpg"
    print("=" * 60)
    print(f"📇 carta_{i}")

    if offline:
        esito = identifica_carta_locale(percorso, indice, reader)
    else:
        # Leggo il nome dalla cella ritagliata (ora ben inquadrata dalle bande reali).
        nomi = leggi_nome_carta(percorso, reader)
        album = nomi_celle is not None
        if album and nomi_celle[i]:
            # album: anteponi i nomi dedotti (confermati dal DB), poi quelli letti
            nomi = list(dict.fromkeys(nomi_celle[i] + nomi))
        print(f"  Nomi candidati: {nomi}")
        # ORB come giudice fine solo in album (sui casi duri olografici); staccate = v1
        esito = identifica_carta(percorso, nomi, reader, lista_sets, usa_orb=album)

    if esito["carta"]:
        distanza, nome_carta, id_carta, local_id = esito["carta"]
    else:
        distanza, nome_carta, id_carta, local_id = None, None, None, None

    prezzo = None
    if nome_carta:
        prezzo = prezzi.prezzo_cardmarket(nome_carta, numero=local_id, cache=cache_prezzi)

    orb = esito.get("orb")   # (inlier, stacco) se confermata da ORB, altrimenti None
    report.append((f"carta_{i}", esito["stato"], nome_carta, id_carta, distanza, prezzo, orb))
    print(f"  → {esito['stato']}: {esito['motivo']}  |  prezzo: {prezzi.formatta_prezzo(prezzo)}")

prezzi.salva_cache(cache_prezzi)

# --- FASE 4: riepilogo finale ---
print("\n" + "=" * 78)
print("RIEPILOGO")
print("=" * 78)

stato_en = {"OK": "OK", "INCERTO": "Uncertain", "FALLITO": "Failed"}

ok = 0
valore_totale = 0.0
for nome_file, stato, nome_carta, id_carta, distanza, prezzo, orb in report:
    simbolo = "✅" if stato == "OK" else "⚠️"
    etichetta = stato_en.get(stato, stato)

    if id_carta and "-" in id_carta:
        set_code, numero = id_carta.rsplit("-", 1)
    else:
        set_code, numero = "—", "—"

    if orb is not None:
        # Confermata da ORB: l'affidabilità riflette la confidenza ORB (inlier +
        # stacco), NON il pHash (che resta alto sulle holo e darebbe una % bassa).
        inlier, stacco = orb
        affidabilita = min(99, 90 + inlier // 50 + stacco // 50)
        aff_txt = f"{affidabilita}%"
    elif distanza is not None:
        if distanza <= 13:
            affidabilita = round(100 - distanza * 0.6)
        else:
            affidabilita = max(20, round(92 - (distanza - 13) * 4))
        aff_txt = f"{affidabilita}%"
    else:
        aff_txt = "—"

    prezzo_txt = prezzi.formatta_prezzo(prezzo)
    if prezzo and prezzo.get("avg"):
        valore_totale += prezzo["avg"]

    print(f"  {simbolo} {nome_file:9} {etichetta:10} {(nome_carta or '—'):20} "
          f"set:{set_code:8} n:{numero:5} aff:{aff_txt:5} {prezzo_txt}")
    if stato == "OK":
        ok += 1

print(f"\nIdentificate con certezza: {ok}/{len(report)}")
print(f"Valore stimato del lotto (somma avg Cardmarket): €{valore_totale:.2f}".replace(".", ","))

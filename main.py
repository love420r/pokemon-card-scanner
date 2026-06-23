import cv2
import sys
import easyocr
from funzioni import (carica_e_ridimensiona, identifica_carta, scarica_sets,
                      trova_rettangoli_ruotati, ritaglia_e_raddrizza, leggi_nome_carta)

# Configurazione
if len(sys.argv) > 1:
    PERCORSO_IMMAGINE = sys.argv[1]
else:
    PERCORSO_IMMAGINE = "carte3.jpg"
    print("Nessun file specificato, uso carte3.jpg (default)\n")
LARGHEZZA_RESIZE = 1000
AREA_MINIMA = 5000
CARTELLA_RITAGLI = "ritagli3"
SOGLIA_OCR = 0.6

# --- FASE 1: dalla foto ai ritagli ---
originale = cv2.imread(PERCORSO_IMMAGINE)
immagine = carica_e_ridimensiona(PERCORSO_IMMAGINE, LARGHEZZA_RESIZE)

if immagine is None:
    print("Impossibile caricare l'immagine. Esco.")
    exit()

rapporto = originale.shape[1] / immagine.shape[1]
rects = trova_rettangoli_ruotati(immagine, AREA_MINIMA)
ritagli = ritaglia_e_raddrizza(originale, rects, rapporto, CARTELLA_RITAGLI)
print(f"Carte trovate e ritagliate: {len(ritagli)}\n")

# --- FASE 2: carica il reader UNA volta ---
print("Caricamento modelli EasyOCR...")
reader = easyocr.Reader(['it', 'en'])
print("Modelli caricati!\n")
print("Scarico la lista dei set...")
lista_sets = scarica_sets()
print(f"Set scaricati: {len(lista_sets)}\n")

# --- FASE 3: identifica ogni carta ---
report = []
for i in range(len(ritagli)):
    percorso = f"{CARTELLA_RITAGLI}/carta_{i}.jpg"
    print("=" * 60)
    print(f"📇 carta_{i}")

    nomi = leggi_nome_carta(percorso, reader)

    if len(nomi) == 0:
        print("  Nessun nome leggibile → tento solo code-first")
        nomi = []

    print(f"  Nomi candidati: {nomi}")
    esito = identifica_carta(percorso, nomi, reader, lista_sets)

    if esito["carta"]:
        distanza = esito["carta"][0]
        nome_carta = esito["carta"][1]
        id_carta = esito["carta"][2]
    else:
        distanza, nome_carta, id_carta = None, None, None
    report.append((f"carta_{i}", esito["stato"], nome_carta, id_carta, distanza))
    print(f"  → {esito['stato']}: {esito['motivo']}")

# --- FASE 4: riepilogo finale ---
print("\n" + "=" * 70)
print("RIEPILOGO")
print("=" * 70)

stato_en = {"OK": "OK", "INCERTO": "Uncertain", "FALLITO": "Failed",
            "AMBIGUO": "Uncertain"}

ok = 0
for nome_file, stato, nome_carta, id_carta, distanza in report:
    simbolo = "✅" if stato == "OK" else "⚠️"
    etichetta = stato_en.get(stato, stato)

    if id_carta and "-" in id_carta:
        set_code, numero = id_carta.rsplit("-", 1)
    else:
        set_code, numero = "—", "—"

    if distanza is not None:
        if distanza <= 13:
            affidabilita = round(100 - distanza * 0.6)      # zona OK: d8→95%
        else:
            affidabilita = max(20, round(92 - (distanza - 13) * 4))  # crolla dopo la soglia
        aff_txt = f"{affidabilita}%"
    else:
        aff_txt = "—"

    print(f"  {simbolo} {nome_file:9} {etichetta:10} {(nome_carta or '—'):20} "
          f"set:{set_code:8} n:{numero:5} aff:{aff_txt:5} [{id_carta or '—'}]")
    if stato == "OK":
        ok += 1

print(f"\nIdentificate con certezza: {ok}/{len(report)}")
import cv2, easyocr
from rapidfuzz import fuzz
from funzioni import carica_e_ridimensiona, cerca_carta_api, deduci_griglia

FOTO = "carte4.jpg"
immagine = carica_e_ridimensiona(FOTO, 1000)
originale = cv2.imread(FOTO)
h, w = immagine.shape[0], immagine.shape[1]

DIZIONARIO = {"cerca","mazzo","aiuto","base","uno","una","che","puoi","giocare",
              "carta","carte","turno","durante","assegnale","strumento","pokemon",
              "pokémon","allenatore","tuo","del","nel","fino","mostrali","oggetto",
              "questo","dei","tre","scelgo","energia","danni","sola","volta","tuoi",
              "thunder","powerful","spark","active","this","does","attacco","rapido"}

reader = easyocr.Reader(['it','en'])
ris = reader.readtext(immagine)

candidati = []
for box, testo, conf in ris:
    t = testo.strip()
    if conf < 0.55 or len(t) < 4 or t.isdigit() or t.isupper():
        continue
    if any(ch.isdigit() for ch in t):
        continue
    if t.lower() in DIZIONARIO:
        continue
    xs=[p[0] for p in box]; ys=[p[1] for p in box]
    candidati.append((t, sum(xs)/4/w, sum(ys)/4/h))

print(f"Candidati dopo filtro locale: {len(candidati)}. Chiedo al DB...\n")

SOGLIA_MATCH = 70   # quanto il nome letto deve somigliare al nome ufficiale
nomi_pos = []
for t, cx, cy in candidati:
    trovati = cerca_carta_api(t)
    if not trovati:
        continue
    nome_ufficiale = trovati[0]["name"]
    # il nome letto deve somigliare DAVVERO al nome trovato (no match laschi)
    somiglianza = fuzz.ratio(t.lower(), nome_ufficiale.lower())
    if somiglianza < SOGLIA_MATCH:
        print(f"  ✗ scarto '{t}' → '{nome_ufficiale}' (somiglianza {somiglianza:.0f})")
        continue
    print(f"  ✅ '{t}' → {nome_ufficiale} ({trovati[0]['id']})  x={cx:.2f} y={cy:.2f}")
    nomi_pos.append((t, cx, cy))

righe, colonne, righe_y, colonne_x = deduci_griglia(nomi_pos, originale.shape[1], originale.shape[0])
print(f"\nGriglia dedotta: {righe} righe × {colonne} colonne = {righe*colonne} carte")
print(f"Righe a y: {[round(y,2) for y in righe_y]}")
print(f"Colonne a x: {[round(x,2) for x in colonne_x]}")
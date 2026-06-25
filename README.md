# Pokémon Card Scanner

🇮🇹 Italiano (sotto) · 🇬🇧 [English version below](#-english)

---

## 🇮🇹 Italiano

Prende in input una foto di carte Pokémon e per ogni carta trova **nome, set,
numero, id**, l'**affidabilità** del riconoscimento e il **prezzo Cardmarket (€)**.
Gestisce sia carte **staccate** sia **pagine di raccoglitore** (carte attaccate in
griglia), scegliendo da solo come ritagliarle.

**Zero setup.** Cloni il repo, lanci `python main.py foto.jpg`, e funziona: il
riconoscimento di default è **online** via TCGdex (sempre aggiornato, nessun indice
da costruire).

### Esempio di output

```
✅ carta_1   OK         Vampeaguzze-ex       set:sv05     n:204   aff:95%   ~€18,40 (trend €19,10)
✅ carta_2   OK         Bisharp              set:bw9      n:73    aff:93%   ~€1,49 (trend €0,90)
⚠️ carta_5   Uncertain  Rayquaza             set:bw6      n:128   aff:56%   ~€44,58 (trend €229,85)

Valore stimato del lotto (somma avg Cardmarket): €48,92
```

### Come funziona

1. **Ritaglio (sceglie da solo):**
   - *Carte staccate* → rilevamento a contorni (pipeline v1: raddrizza ogni carta).
   - *Carte attaccate / album* → quando i contorni non bastano, **deduce la griglia
     dai NOMI** che l'OCR legge: ogni nome di Pokémon valido = una carta, e la sua
     posizione (x,y) dice dove sta. Righe dal raggruppamento delle y, colonne dal
     raggruppamento delle x. Niente bisogno di scrivere la griglia a mano.
2. **Riconoscimento (online di default):** OCR del nome → ricerca su TCGdex →
   confronto immagine con **pHash** sui candidati scaricati al volo.
3. **Prezzo:** cerca il prezzo Cardmarket (€) su pokemontcg.io e mostra il valore
   totale stimato del lotto.

### Come si usa

Installa le dipendenze e lancia: nient'altro.

```
pip install -r requirements.txt
python main.py mia_foto.jpg              # carte staccate o album: decide da solo
```

Se non passi nessun file, usa `carte3.jpg` come default.

**Override manuale della griglia** (solo se l'auto-deduzione sbaglia il conteggio):

```
python main.py pagina.jpg --griglia 3x3
```

### Prezzi (API key opzionale)

I prezzi arrivano da [pokemontcg.io](https://pokemontcg.io). Senza chiave funziona
ma con rate limit bassi; con una chiave gratuita imposti la variabile d'ambiente
`POKEMONTCGIO_KEY`. I prezzi vengono messi in cache giornaliera in `prezzi_cache.json`.

### Modalità OFFLINE (avanzata, opzionale)

Per chi vuole velocità e indipendenza dalla rete in fase di riconoscimento, c'è un
**database pHash locale**. È un'opzione **avanzata e opzionale**: va costruita una
volta (può richiedere tempo) e non è inclusa nel repo (invecchia ed è parziale).

```
python costruisci_db.py --set swsh3 sv03     # costruisci l'indice dei tuoi set
python costruisci_db.py --all                # oppure tutti (lungo)
python main.py foto.jpg --offline            # usa il DB locale invece di TCGdex
```

### Limiti noti

- **Carte attaccate / album:** la griglia viene dedotta correttamente, ma il
  riconoscimento per cella è ancora limitato dalla **rifinitura del ritaglio**
  (busta protettiva e bordo cella alzano la distanza pHash). Prossimo blocco di lavoro.
- Le **carte vecchie** hanno immagini di riferimento di bassa qualità → affidabilità più bassa.

### Tecnologie

OpenCV · EasyOCR · pHash (imagehash) · RapidFuzz · NumPy ·
API TCGdex (identificazione) · API pokemontcg.io (prezzi Cardmarket) ·
SQLite (solo per la modalità offline opzionale)

---

## 🇬🇧 English

Takes a photo of Pokémon cards and, for each card, finds its **name, set, number,
id**, the recognition **confidence** and the **Cardmarket price (€)**. Handles both
**loose** cards and **binder pages** (cards attached in a grid), deciding on its own
how to crop them.

**Zero setup.** Clone the repo, run `python main.py photo.jpg`, and it works: default
recognition is **online** via TCGdex (always up to date, no index to build).

### Example output

```
✅ carta_1   OK         Vampeaguzze-ex       set:sv05     n:204   aff:95%   ~€18,40 (trend €19,10)
✅ carta_2   OK         Bisharp              set:bw9      n:73    aff:93%   ~€1,49 (trend €0,90)
⚠️ carta_5   Uncertain  Rayquaza             set:bw6      n:128   aff:56%   ~€44,58 (trend €229,85)

Estimated lot value (sum of Cardmarket avg): €48,92
```

### How it works

1. **Cropping (auto):**
   - *Loose cards* → contour detection (v1 pipeline: each card straightened).
   - *Attached / binder cards* → when contours aren't enough, it **deduces the grid
     from the NAMES** the OCR reads: each valid Pokémon name = a card, and its (x,y)
     position says where it is. Rows from grouping the y's, columns from grouping the
     x's. No need to type the grid by hand.
2. **Recognition (online by default):** name OCR → TCGdex search → image comparison
   with **pHash** on candidates downloaded on the fly.
3. **Price:** looks up the Cardmarket price (€) on pokemontcg.io and shows the
   estimated total lot value.

### How to use it

Install dependencies and run: nothing else.

```
pip install -r requirements.txt
python main.py my_photo.jpg              # loose cards or binder: it decides
```

If you don't pass any file, it uses `carte3.jpg` as the default.

**Manual grid override** (only if the auto-deduction miscounts):

```
python main.py page.jpg --griglia 3x3
```

### Prices (optional API key)

Prices come from [pokemontcg.io](https://pokemontcg.io). It works without a key but
with low rate limits; with a free key, set the `POKEMONTCGIO_KEY` environment
variable. Prices are cached daily in `prezzi_cache.json`.

### OFFLINE mode (advanced, optional)

For those who want speed and no network during recognition, there's a **local pHash
database**. It's an **advanced, optional** feature: it must be built once (can take a
while) and is not included in the repo (it ages and is partial).

```
python costruisci_db.py --set swsh3 sv03     # build the index for your sets
python costruisci_db.py --all                # or all of them (slow)
python main.py photo.jpg --offline           # use the local DB instead of TCGdex
```

### Known limitations

- **Attached / binder cards:** cells are cut along the **real card edges** found by
  the projection profile (`rileva_griglia`), reconciled with the row/column count
  deduced from the names, then refined (`rifinisci_cella` trims sleeve/dividers).
  This makes every card name readable (uniform slicing cut some names off). **Holo /
  rainbow secret-rares keep a high pHash distance regardless of the crop** (the foil
  diverges from the flat reference scan — an intrinsic pHash floor), so they stay
  "uncertain" unless their set code is readable (see the tie-breaker below).
- **Set-code tie-breaker:** when the name is known but the pHash stays high, the
  card is confirmed OK if the read code (`number/total`) matches a candidate's
  `localId` **and** its set total. This rescues holo cards **when the code is
  readable** — on tightly-packed binder pages the holo codes are often tiny over
  foil and cut at the pocket edge, so it doesn't always fire.
- **ORB tie-breaker (binder pages):** for the hard holo cells, when the pHash stays
  > 13, ORB feature matching re-ranks the name's candidates and confirms OK only if
  the best candidate clears an inlier threshold **and** separates from the second.
  ORB matches local keypoints (robust to the foil that blinds pHash), so it picks
  the right printing — e.g. it recovers the rainbow Articuno GX and corrects a pHash
  mismatch on Drampa GX. It only runs on binder pages (the loose-card v1 path is
  untouched) and downloads full-res references on first use (cached afterwards).
- **Old cards** have low-quality reference images → lower confidence.

### Tech stack

OpenCV · EasyOCR · pHash (imagehash) · RapidFuzz · NumPy ·
TCGdex API (identification) · pokemontcg.io API (Cardmarket prices) ·
SQLite (only for the optional offline mode)

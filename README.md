# Pokémon Card Scanner

🇮🇹 Italiano (sotto) · 🇬🇧 [English version below](#-english)

---

## 🇮🇹 Italiano

Prende in input una foto di carte Pokémon (staccate tra loro) e per ogni carta trova
**nome, set, numero, id** e l'**affidabilità** con cui è stata riconosciuta.

### Esempio di output

```
✅ carta_1   OK         Vampeaguzze-ex       set:sv05     n:204   aff:95%   [sv05-204]
✅ carta_2   OK         Bisharp              set:bw9      n:73    aff:93%   [bw9-73]
⚠️ carta_5   Uncertain  Rayquaza             set:bw6      n:128   aff:56%   [bw6-128]
```

### Come funziona

1. **Ritaglia** ogni carta trovata nella foto e la **raddrizza** (le carte storte
   vengono rimesse dritte, perché il confronto immagine non perdona la rotazione).
2. **Legge il nome** con OCR, concentrandosi sulla fascia alta della carta dove sta
   il nome (così non lo confonde con gli attacchi).
3. Con un **match fuzzy** individua il nome giusto anche se l'OCR lo storpia un po',
   e cerca le carte con quel nome sull'**API di TCGdex**.
4. **Confronta le immagini** dei candidati con la foto usando **pHash**: vince la carta
   più vicina. La distanza diventa l'affidabilità.
5. Da TCGdex ricava **set, numero e id** della carta.

### Come si usa

Installa le dipendenze:

```
pip install -r requirements.txt
```

Lancia, passando la tua foto (formato jpg):

```
python main.py mia_foto.jpg
```

Se non passi nessun file, usa `carte3.jpg` come default.

### Limiti noti

- Le **carte vecchie** hanno immagini di riferimento di bassa qualità → affidabilità più bassa.
- Le **energie** e le carte **non in italiano o inglese** (es. giapponesi) non vengono
  riconosciute dal nome, perché l'OCR è impostato su italiano/inglese.

### Tecnologie

OpenCV · EasyOCR · pHash (imagehash) · RapidFuzz · API TCGdex

---

## 🇬🇧 English

Takes a photo of Pokémon cards (separated from each other) as input and, for each card,
finds its **name, set, number, id** and the **confidence** with which it was recognized.

### Example output

```
✅ carta_1   OK         Vampeaguzze-ex       set:sv05     n:204   aff:95%   [sv05-204]
✅ carta_2   OK         Bisharp              set:bw9      n:73    aff:93%   [bw9-73]
⚠️ carta_5   Uncertain  Rayquaza             set:bw6      n:128   aff:56%   [bw6-128]
```

### How it works

1. **Crops** each card found in the photo and **straightens** it (tilted cards are
   rotated upright, because image comparison does not forgive rotation).
2. **Reads the name** with OCR, focusing on the top strip of the card where the name is
   (so it doesn't confuse it with the attacks).
3. Uses **fuzzy matching** to find the right name even if the OCR garbles it a bit,
   then searches for cards with that name on the **TCGdex API**.
4. **Compares the images** of the candidates against the photo using **pHash**: the
   closest card wins. The distance becomes the confidence score.
5. Retrieves **set, number and id** of the card from TCGdex.

### How to use it

Install the dependencies:

```
pip install -r requirements.txt
```

Run it, passing your photo (jpg format):

```
python main.py my_photo.jpg
```

If you don't pass any file, it uses `carte3.jpg` as the default.

### Known limitations

- **Old cards** have low-quality reference images → lower confidence.
- **Energy cards** and cards **not in Italian or English** (e.g. Japanese) are not
  recognized by name, because the OCR is set to Italian/English.

### Tech stack

OpenCV · EasyOCR · pHash (imagehash) · RapidFuzz · TCGdex API
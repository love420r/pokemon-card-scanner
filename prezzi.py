"""
Prezzi delle carte da Cardmarket (in €), via API pokemontcg.io.

TCGdex (che usiamo per identificare) è ottimo per i metadati ma debole sui
prezzi; pokemontcg.io invece espone i prezzi Cardmarket per ogni carta.
Qui cerchiamo la carta per nome (+ numero) e leggiamo il campo cardmarket.

Cache: i prezzi cambiano nel tempo, quindi NON stanno nel DB statico. Li
chiediamo al volo e li teniamo in un file di cache valido per il giorno corrente.

API key (opzionale ma consigliata, gratuita su https://pokemontcg.io):
    imposta la variabile d'ambiente POKEMONTCGIO_KEY per alzare i rate limit.
"""

import os
import json
import datetime
import requests
from rapidfuzz import fuzz

API = "https://api.pokemontcg.io/v2/cards"
CACHE_PATH = "prezzi_cache.json"


def carica_cache(percorso=CACHE_PATH):
    if os.path.exists(percorso):
        try:
            with open(percorso, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def salva_cache(cache, percorso=CACHE_PATH):
    try:
        with open(percorso, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def prezzo_cardmarket(nome, numero=None, cache=None, api_key=None):
    """
    Ritorna i prezzi Cardmarket (€) della carta, o None se non trovati.
    Struttura ritornata:
        {avg, trend, low, valuta: "EUR", url, aggiornato}
    nome   = nome della carta (in inglese: l'API è in inglese)
    numero = numero stampato sulla carta (migliora il match), es. "136"
    cache  = dict riusabile tra chiamate (vedi carica_cache/salva_cache)
    """
    if api_key is None:
        api_key = os.environ.get("POKEMONTCGIO_KEY")

    oggi = datetime.date.today().isoformat()
    chiave = f"{nome}|{numero}"
    if cache is not None and chiave in cache and cache[chiave].get("data") == oggi:
        return cache[chiave]["prezzo"]

    # Costruisci la query Lucene di pokemontcg.io
    q = f'name:"{nome}"'
    if numero:
        q += f' number:"{numero}"'
    headers = {"X-Api-Key": api_key} if api_key else {}

    try:
        r = requests.get(API, params={"q": q, "pageSize": 10}, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        dati = r.json().get("data", [])
    except requests.exceptions.RequestException:
        return None

    if not dati:
        return None

    # Scegli il candidato col nome più simile (l'OCR/DB può variare un po')
    dati.sort(key=lambda c: -fuzz.ratio(nome.lower(), c.get("name", "").lower()))
    carta = dati[0]

    cm = carta.get("cardmarket") or {}
    prezzi = cm.get("prices") or {}
    if not prezzi:
        return None

    risultato = {
        "avg": prezzi.get("averageSellPrice"),
        "trend": prezzi.get("trendPrice"),
        "low": prezzi.get("lowPrice"),
        "valuta": "EUR",
        "url": cm.get("url"),
        "aggiornato": cm.get("updatedAt"),
    }

    if cache is not None:
        cache[chiave] = {"data": oggi, "prezzo": risultato}
    return risultato


def formatta_prezzo(prezzo):
    # Stringa breve per il report, es. "~€12,40 (trend €13,10)". "—" se assente.
    if not prezzo:
        return "—"
    avg = prezzo.get("avg")
    trend = prezzo.get("trend")
    if avg is None and trend is None:
        return "—"
    base = f"~€{avg:.2f}".replace(".", ",") if avg is not None else "—"
    if trend is not None:
        base += f" (trend €{trend:.2f})".replace(".", ",")
    return base

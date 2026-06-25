"""
Costruisce l'indice locale di pHash (una tantum, resumabile).

Esempi:
    python costruisci_db.py --set swsh3 sv03      # solo questi set (consigliato per iniziare)
    python costruisci_db.py --all                 # TUTTI i set (pesante: ore di download)
    python costruisci_db.py --set sv03 --lingua it

Puoi rilanciarlo quando vuoi: salta le carte già indicizzate e aggiunge solo le nuove.
"""

import argparse
import hash_db


def main():
    p = argparse.ArgumentParser(description="Costruisce il DB locale di pHash delle carte Pokémon.")
    p.add_argument("--set", nargs="+", metavar="SET_ID",
                   help="uno o più set_id da indicizzare (es. swsh3 sv03)")
    p.add_argument("--all", action="store_true",
                   help="indicizza TUTTI i set (lungo)")
    p.add_argument("--lingua", default="en",
                   help="lingua dell'API TCGdex (default: en — l'artwork è uguale tra le lingue)")
    p.add_argument("--db", default=hash_db.DB_PATH,
                   help=f"percorso del file DB (default: {hash_db.DB_PATH})")
    p.add_argument("--workers", type=int, default=12,
                   help="download paralleli (default: 12)")
    args = p.parse_args()

    if not args.all and not args.set:
        p.error("specifica --set <id ...> oppure --all")

    sets = None if args.all else args.set
    hash_db.costruisci(sets=sets, lingua=args.lingua, db_path=args.db, workers=args.workers)


if __name__ == "__main__":
    main()

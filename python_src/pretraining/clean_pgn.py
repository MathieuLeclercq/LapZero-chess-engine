
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import os
import re
import hashlib

# Dossiers d'entrée et de sortie
INPUT_DIR = r"..\training_data\raw_pgns"
OUTPUT_DIR = r"..\training_data\clean_pgns"


def sanitize_filename(name):
    """Nettoie les chaînes de caractères pour en faire des noms de fichiers valides."""
    # Remplace les points, virgules, espaces et caractères bizarres par des underscores
    clean = re.sub(r'[^A-Za-z0-9]', '_', name)
    # Enlève les underscores multiples
    return re.sub(r'_+', '_', clean).strip('_')


def get_games(filepath):
    """Générateur qui lit un gros fichier PGN et renvoie les parties une par une."""
    game_lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('[Event ') and game_lines:
                yield "".join(game_lines)
                game_lines = []
            game_lines.append(line)
        if game_lines:
            yield "".join(game_lines)


def process_pgns():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_hashes = set()
    total_games = 0
    saved_games = 0
    duplicates = 0

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".pgn"):
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        print(f"Traitement de {filename}...")

        for game in get_games(filepath):
            total_games += 1
            if not game.strip():
                continue

            # --- 1. Extraction des métadonnées ---
            white_match = re.search(r'\[White "(.*?)"\]', game)
            black_match = re.search(r'\[Black "(.*?)"\]', game)
            date_match = re.search(r'\[Date "(.*?)"\]', game)

            white = white_match.group(1) if white_match else "Unknown"
            black = black_match.group(1) if black_match else "Unknown"
            date = date_match.group(1) if date_match else "Unknown"

            # --- 2. Détection des doublons (Hashing) ---
            # On retire tous les tags [...] pour ne garder que la séquence de coups
            moves_text = re.sub(r'\[.*?\]', '', game)
            # On retire les espaces et sauts de ligne pour avoir une empreinte stricte
            moves_clean = re.sub(r'\s+', '', moves_text)

            # L'empreinte unique d'une partie : Blancs + Noirs + Coups joués
            unique_string = f"{white}{black}{moves_clean}"
            game_hash = hashlib.md5(unique_string.encode('utf-8')).hexdigest()

            if game_hash in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(game_hash)

            # --- 3. Formatage du nom de fichier ---
            safe_white = sanitize_filename(white)
            safe_black = sanitize_filename(black)
            safe_date = sanitize_filename(date)

            base_name = f"{safe_white}_vs_{safe_black}_{safe_date}"
            out_name = f"{base_name}.pgn"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            # Gestion des collisions (ex: tie-breaks rapides, 2 parties le même jour)
            counter = 1
            while os.path.exists(out_path):
                out_name = f"{base_name}_{counter}.pgn"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                counter += 1

            # --- 4. Sauvegarde ---
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(game.strip() + "\n\n")

            saved_games += 1

    print("\n--- Bilan de l'extraction ---")
    print(f"Parties totales lues : {total_games}")
    print(f"Doublons éliminés    : {duplicates}")
    print(f"Fichiers sauvegardés : {saved_games}")


if __name__ == "__main__":
    process_pgns()
    
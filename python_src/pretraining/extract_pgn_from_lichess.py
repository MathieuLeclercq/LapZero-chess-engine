
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import zstandard as zstd
import io

INPUT_ZST = r"..\training_data\lichess_dataset\lichess_db_standard_rated_2026-02.pgn.zst"
OUTPUT_PGN = r"..\training_data\lichess_dataset\top_games_quality.pgn"
MIN_ELO = 1800
TARGET_GAMES = 1_000_000


def filter_grandmaster_games():
    count = 0
    current_game = []
    white_elo, black_elo = 0, 0

    print(f"Extraction des parties où les deux joueurs sont >= {MIN_ELO}...")

    with open(INPUT_ZST, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')

            with open(OUTPUT_PGN, 'w', encoding='utf-8') as f_out:
                for line in text_stream:
                    current_game.append(line)

                    if line.startswith('[WhiteElo "'):
                        white_elo = int(line.split('"')[1])
                    elif line.startswith('[BlackElo "'):
                        black_elo = int(line.split('"')[1])

                    # Fin d'une partie (ligne vide après les coups)
                    if line.strip() == "" and len(current_game) > 10:
                        if white_elo >= MIN_ELO and black_elo >= MIN_ELO:
                            f_out.write("".join(current_game) + "\n")
                            count += 1
                            if count % 10000 == 0:
                                print(f"Parties extraites : {count}")

                        # Reset
                        current_game = []
                        white_elo, black_elo = 0, 0

                        if count >= TARGET_GAMES:
                            break

    print(f"Terminé : {count} parties GMs sauvegardées.")


if __name__ == "__main__":
    filter_grandmaster_games()

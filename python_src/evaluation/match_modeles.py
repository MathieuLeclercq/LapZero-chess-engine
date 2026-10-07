"""Match entre deux modeles ONNX, par paires d'ouvertures a couleurs inversees.

Chaque paire tire une ouverture de quelques demi-coups, puis la joue deux fois :
une fois avec le modele A aux Blancs, une fois avec le modele B. Ce schema
neutralise l'avantage du trait et le hasard de l'ouverture, et l'intervalle de
confiance est calcule sur les paires, ce qui le resserre nettement par rapport a
des parties independantes.

Apres l'ouverture, chaque camp joue le coup le plus visite de sa recherche, sans
tirage : le match mesure la force et non la diversite.

Usage, depuis python_src :

    python match_modeles.py --modele-a checkpoints_onnx/A.onnx \\
        --modele-b checkpoints_onnx/B.onnx --paires 100 --simulations 400
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPERTOIRE = _runtime.PYTHON_ROOT

# Memes reglages que le tournoi et l'ancrage Stockfish.
TAILLE_TT = 131071
TAILLE_LOT = 8
VIRTUAL_LOSS = 2
C_PUCT = 1.4


# ============================================================
#                     STATISTIQUES
# ============================================================
def elo_depuis_score(score):
    """Ecart Elo correspondant a un score moyen par partie, borne pour rester fini."""
    s = min(max(score, 1e-3), 1 - 1e-3)
    # + 0.0 evite d'afficher -0 pour un score de 0,5 exactement.
    return -400.0 * math.log10(1.0 / s - 1.0) + 0.0


@dataclass
class Bilan:
    paires: int
    parties: int
    victoires_b: int
    nulles: int
    defaites_b: int
    score_b: float
    elo_b: float
    elo_bas: float
    elo_haut: float


def bilan_du_match(scores_paires, victoires_b, nulles, defaites_b):
    """Bilan du point de vue du modele B.

    scores_paires contient, pour chaque paire, les points de B sur ses deux
    parties (0 a 2). L'ecart-type est pris sur les paires, pas sur les
    parties, puisque les deux parties d'une paire partagent leur ouverture.
    """
    n = len(scores_paires)
    if n == 0:
        raise ValueError("bilan_du_match : aucune paire jouee")
    moyenne = sum(scores_paires) / (2 * n)
    if n > 1:
        erreur = statistics.stdev(scores_paires) / (2 * math.sqrt(n))
    else:
        erreur = 0.5
    return Bilan(
        paires=n, parties=2 * n,
        victoires_b=victoires_b, nulles=nulles, defaites_b=defaites_b,
        score_b=moyenne,
        elo_b=elo_depuis_score(moyenne),
        elo_bas=elo_depuis_score(moyenne - 1.96 * erreur),
        elo_haut=elo_depuis_score(moyenne + 1.96 * erreur),
    )


# ============================================================
#                     PARTIES (processus travailleurs)
# ============================================================
_MOTEUR = {}


def initialiser_travailleur(chemin_a, chemin_b):
    """Charge les deux reseaux une seule fois par processus."""
    os.add_dll_directory(str(REPERTOIRE))
    sys.path.insert(0, str(REPERTOIRE))
    import chess_engine
    _MOTEUR["ce"] = chess_engine
    _MOTEUR["A"] = chess_engine.ONNXEvaluator(str(chemin_a), False)
    _MOTEUR["B"] = chess_engine.ONNXEvaluator(str(chemin_b), False)


def _nouvelle_recherche(cle):
    ce = _MOTEUR["ce"]
    mcts = ce.MCTS(_MOTEUR[cle], tt_size=TAILLE_TT)
    mcts.set_tuning(VIRTUAL_LOSS)
    return mcts


def _jouer(board, coords):
    """Joue un coup en mettant a jour l'etat de la partie ; rend (UCI, SAN).

    move_piece_uci ne teste pas la fin de partie, c'est voulu pour les bancs
    qui rejouent des historiques : ici un mat doit terminer la partie.
    """
    from core.lib import coords_to_uci, move_to_san
    ce = _MOTEUR["ce"]
    uci = coords_to_uci(*coords)
    san = move_to_san(board, *coords)
    if not board.move_piece(*coords):
        raise RuntimeError(f"coup refuse : {uci}")
    if board.game_state == ce.GameState.CHECKMATE:
        san += "#"
    elif board.is_in_check():
        san += "+"
    return uci, san


def _jouer_index(board, index):
    return _jouer(board, board.decode_move_index(int(index)))


def _jouer_uci(board, uci):
    from core.lib import parse_uci_to_coords
    return _jouer(board, parse_uci_to_coords(uci))


def tirer_ouverture(graine, plies, simulations):
    """Ouverture tiree par le modele A, a temperature 1, reproductible par sa graine."""
    ce = _MOTEUR["ce"]
    rng = np.random.default_rng(graine)
    mcts = _nouvelle_recherche("A")
    board = ce.Chessboard()
    board.set_startup_pieces()
    coups = []
    for _ in range(plies):
        if board.game_state != ce.GameState.ONGOING:
            break
        pi = np.asarray(mcts.mcts_search(board, simulations, C_PUCT, False, TAILLE_LOT),
                        dtype=np.float64)
        coups.append(_jouer_index(board, rng.choice(len(pi), p=pi / pi.sum()))[0])
    return coups


def jouer_partie(ouverture, cle_blancs, simulations, plafond_plies):
    """Joue une partie depuis l'ouverture ; rend (resultat PGN, terminaison, coups SAN)."""
    ce = _MOTEUR["ce"]
    board = ce.Chessboard()
    board.set_startup_pieces()
    coups = [_jouer_uci(board, uci)[1] for uci in ouverture]
    cle_noirs = "B" if cle_blancs == "A" else "A"
    recherches = {"A": _nouvelle_recherche("A"), "B": _nouvelle_recherche("B")}

    while board.game_state == ce.GameState.ONGOING and len(coups) < plafond_plies:
        cle = cle_blancs if board.turn == ce.Color.WHITE else cle_noirs
        pi = recherches[cle].mcts_search(board, simulations, C_PUCT, False, TAILLE_LOT)
        coups.append(_jouer_index(board, int(np.argmax(pi)))[1])

    etat = board.game_state
    if etat == ce.GameState.CHECKMATE:
        resultat = "0-1" if board.turn == ce.Color.WHITE else "1-0"
        terminaison = "mat"
    elif etat == ce.GameState.ONGOING:
        resultat, terminaison = "1/2-1/2", "plafond"
    else:
        resultat = "1/2-1/2"
        terminaison = {
            ce.GameState.STALEMATE: "pat",
            ce.GameState.DRAW_REPETITION: "repetition",
            ce.GameState.DRAW_50_MOVES: "50 coups",
            ce.GameState.DRAW_INSUFF_MATERIAL: "materiel",
        }.get(etat, str(etat))
    return resultat, terminaison, coups


def jouer_paire(args):
    numero, graine, plies_ouverture, sims_ouverture, simulations, plafond = args
    ouverture = tirer_ouverture(graine, plies_ouverture, sims_ouverture)
    parties = []
    for cle_blancs in ("A", "B"):
        resultat, terminaison, coups = jouer_partie(ouverture, cle_blancs, simulations, plafond)
        parties.append({"blancs": cle_blancs, "resultat": resultat,
                        "terminaison": terminaison, "coups": coups})
    return numero, ouverture, parties


# ============================================================
#                     ORCHESTRATION
# ============================================================
def points_de_b(partie):
    if partie["resultat"] == "1/2-1/2":
        return 0.5
    blancs_gagnent = partie["resultat"] == "1-0"
    return 1.0 if blancs_gagnent == (partie["blancs"] == "B") else 0.0


def pgn(partie, noms, numero, ouverture):
    blanc = noms[partie["blancs"]]
    noir = noms["B" if partie["blancs"] == "A" else "A"]
    texte = (f'[Event "Match {noms["A"]} contre {noms["B"]}"]\n'
             f'[Round "{numero}"]\n[White "{blanc}"]\n[Black "{noir}"]\n'
             f'[Result "{partie["resultat"]}"]\n'
             f'[Termination "{partie["terminaison"]}"]\n'
             f'[Opening "{" ".join(ouverture)}"]\n\n')
    corps = []
    for i, san in enumerate(partie["coups"]):
        if i % 2 == 0:
            corps.append(f"{i // 2 + 1}.")
        corps.append(san)
    return texte + " ".join(corps) + f" {partie['resultat']}\n\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modele-a", type=Path, required=True, help="modele de reference")
    parser.add_argument("--modele-b", type=Path, required=True, help="modele candidat")
    parser.add_argument("--paires", type=int, default=100,
                        help="nombre d'ouvertures, chacune jouee deux fois")
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--travailleurs", type=int, default=16)
    parser.add_argument("--plies-ouverture", type=int, default=8)
    parser.add_argument("--simulations-ouverture", type=int, default=32)
    parser.add_argument("--plafond-plies", type=int, default=300)
    parser.add_argument("--graine", type=int, default=2026)
    parser.add_argument("--sortie", type=Path, default=REPERTOIRE.parent / "out" / "matchs")
    args = parser.parse_args()

    for chemin in (args.modele_a, args.modele_b):
        if not chemin.is_file():
            parser.error(f"modele introuvable : {chemin}")

    noms = {"A": args.modele_a.stem, "B": args.modele_b.stem}
    horodatage = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    args.sortie.mkdir(parents=True, exist_ok=True)
    chemin_pgn = args.sortie / f"match_{horodatage}.pgn"
    chemin_bilan = args.sortie / f"match_{horodatage}.json"

    taches = [(k + 1, args.graine + k, args.plies_ouverture, args.simulations_ouverture,
               args.simulations, args.plafond_plies) for k in range(args.paires)]

    print(f"A = {noms['A']}\nB = {noms['B']}")
    print(f"{args.paires} paires, {2 * args.paires} parties, {args.simulations} simulations, "
          f"{args.travailleurs} processus CPU\n")

    scores_paires, ouvertures = [], set()
    decompte = {"V": 0, "N": 0, "D": 0}
    terminaisons = {}
    debut = time.perf_counter()
    try:
        with mp.Pool(args.travailleurs, initializer=initialiser_travailleur,
                     initargs=(args.modele_a.resolve(), args.modele_b.resolve())) as pool, \
                open(chemin_pgn, "w", encoding="utf-8") as fichier_pgn:
            for numero, ouverture, parties in pool.imap_unordered(jouer_paire, taches):
                ouvertures.add(tuple(ouverture))
                points = 0.0
                for partie in parties:
                    p = points_de_b(partie)
                    points += p
                    decompte["V" if p == 1 else "N" if p == 0.5 else "D"] += 1
                    terminaisons[partie["terminaison"]] = terminaisons.get(partie["terminaison"], 0) + 1
                    fichier_pgn.write(pgn(partie, noms, numero, ouverture))
                fichier_pgn.flush()
                scores_paires.append(points)
                b = bilan_du_match(scores_paires, decompte["V"], decompte["N"], decompte["D"])
                ecoule = time.perf_counter() - debut
                reste = ecoule / len(scores_paires) * (args.paires - len(scores_paires))
                print(f"paire {len(scores_paires):3d}/{args.paires}  B {points:.1f}/2  |  "
                      f"B : +{b.victoires_b} ={b.nulles} -{b.defaites_b}  score {b.score_b:.3f}  "
                      f"Elo {b.elo_b:+.0f} [{b.elo_bas:+.0f} ; {b.elo_haut:+.0f}]  "
                      f"|  reste ~{reste / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        print("\nInterruption : bilan sur les paires terminees.")

    if not scores_paires:
        return
    b = bilan_du_match(scores_paires, decompte["V"], decompte["N"], decompte["D"])
    duree = time.perf_counter() - debut
    resume = {"modele_a": noms["A"], "modele_b": noms["B"], "simulations": args.simulations,
              "plies_ouverture": args.plies_ouverture, "graine": args.graine,
              "ouvertures_distinctes": len(ouvertures), "terminaisons": terminaisons,
              "duree_s": round(duree), **asdict(b)}
    chemin_bilan.write_text(json.dumps(resume, indent=1), encoding="utf-8")

    print(f"\n{'=' * 60}\nBilan pour B ({noms['B']}) contre A ({noms['A']})")
    print(f"  {b.parties} parties : +{b.victoires_b} ={b.nulles} -{b.defaites_b}, score {b.score_b:.3f}")
    print(f"  Ecart Elo : {b.elo_b:+.0f}, intervalle a 95 % [{b.elo_bas:+.0f} ; {b.elo_haut:+.0f}]")
    print(f"  Fins de partie : {terminaisons}")
    print(f"  Ouvertures distinctes : {len(ouvertures)} sur {b.paires}")
    print(f"  Duree : {duree / 60:.1f} min\n  PGN : {chemin_pgn}\n  Bilan : {chemin_bilan}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

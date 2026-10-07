"""Tests des sources externes du banc de positions.

Le PGN de reference vit dans tests/data/position_bench/broadcasts.pgn : deux
parties valides (avec commentaires, variante et un joueur BOT) et une partie
par cause de rejet. Aucun acces reseau : le telechargement est injecte.
"""
import hashlib
import os
import random
import re
import sys
from pathlib import Path
from typing import cast

import chess
import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine
import evaluation.position_bench_sources as position_bench_sources
from evaluation.position_bench_metrics import Position, SourceInfo, validate_position
from evaluation.position_bench_sources import (
    DonneesInvalides,
    barre_progression,
    dedupliquer_candidats,
    download_archive,
    extract_candidates,
    formater_progression,
    position_identity,
    read_games,
    replay_position,
)

BROADCASTS = RACINE / "tests" / "data" / "position_bench" / "broadcasts.pgn"

SOURCE: SourceInfo = {
    "source_id": "lichess-broadcasts-2026-08",
    "url": ("https://database.lichess.org/broadcasts/"
            "lichess-broadcasts-2026-08.pgn.zst"),
    "month": "2026-08",
    "archive_sha256": "c" * 64,
    "license": "CC BY-SA 4.0",
    "attribution": "Lichess",
}

REJETS_ATTENDUS = {
    "elo_insuffisant": 1,
    "elo_absent": 1,
    "variante_non_standard": 1,
    "depart_non_standard": 1,
    "partie_illegale": 1,
    "resultat_incomplet": 1,
    "date_ancienne": 1,
    "partie_trop_courte": 1,
    "identite_evenement": 1,
    "date_hors_mois": 1,
}


def _parties():
    return list(read_games(BROADCASTS, SOURCE))


def _resume(candidats):
    return [(c["position_id"], c["game_id"], c["ply"], c["phase"], c["fen"],
             tuple(c["moves_uci"]), tuple(c["legal_indices"]))
            for c in candidats]


def _plateau(coups_uci):
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    for coup in coups_uci:
        assert board.move_piece_uci(coup), coup
    return board


# ============================================================
#                     LECTURE DES ARCHIVES
# ============================================================

def test_read_games_lit_toutes_les_parties_du_pgn():
    parties = _parties()

    assert len(parties) == 11
    assert parties[0].headers["Event"] == "Test Broadcast A"
    assert getattr(parties[0], "source_info", None) == SOURCE


def test_read_games_lit_le_zst_comme_le_pgn(tmp_path):
    import zstandard as zstd

    destination = tmp_path / "broadcasts.pgn.zst"
    destination.write_bytes(
        zstd.ZstdCompressor().compress(BROADCASTS.read_bytes()))

    parties = list(read_games(destination, SOURCE))

    assert len(parties) == 11
    assert [p.headers["Event"] for p in parties] == [
        p.headers["Event"] for p in _parties()]


# ============================================================
#                  EXTRACTION ET REJETS
# ============================================================

def test_extraction_compte_chaque_cause_de_rejet():
    compteurs = {}

    candidats = extract_candidates(_parties(), SOURCE, compteurs=compteurs)

    assert compteurs == REJETS_ATTENDUS
    assert len(candidats) == 4


def test_extract_candidates_signale_la_progression():
    appels = []

    candidats = extract_candidates(_parties(), SOURCE, progress=appels.append,
                                   progress_every=1)

    assert len(appels) == 11
    assert appels[-1]["parties_lues"] == 11
    assert appels[-1]["rejets"] == (
        sum(REJETS_ATTENDUS.values()) - REJETS_ATTENDUS["date_hors_mois"])
    assert appels[-1]["candidats"] == len(candidats)


def test_les_candidats_respectent_le_contrat_et_l_ordre_canonique():
    candidats = extract_candidates(_parties(), SOURCE)

    for candidat in candidats:
        validate_position(candidat)

    couples = [(c["game_id"], c["phase"]) for c in candidats]
    assert len(couples) == len(set(couples))
    assert {c["game_id"] for c in candidats} == {"aaaaaaaa", "bbbbbbbb"}
    assert len({c["position_id"] for c in candidats}) == len(candidats)
    cles = [(c["game_id"], c["ply"], c["source_id"]) for c in candidats]
    assert cles == sorted(cles)
    for candidat in candidats:
        assert candidat["event_id"]
        assert candidat["legal_indices"] == sorted(
            set(candidat["legal_indices"]))
        assert candidat["game_date"].startswith("2026-")


def test_le_rejeu_retrouve_la_position_et_l_historique():
    for candidat in extract_candidates(_parties(), SOURCE):
        board = replay_position(candidat)

        assert board.to_fen() == candidat["fen"]
        assert len(candidat["moves_uci"]) == candidat["ply"]
        assert len(board.get_board_history()) == candidat["ply"] + 1


def test_rejeu_garde_la_case_en_passant():
    ref = chess.Board()
    ref.push_uci("e2e4")
    record = {"start_fen": chess.STARTING_FEN, "moves_uci": ["e2e4"],
              "fen": ref.fen(en_passant="fen")}

    board = replay_position(record)

    assert board.to_fen() == record["fen"]


def test_rejeu_refuse_un_historique_illegal():
    with pytest.raises(DonneesInvalides) as info:
        replay_position({"start_fen": chess.STARTING_FEN,
                         "moves_uci": ["e2e4", "e2e4"]})

    assert info.value.cause == "historique_illegal"


def test_meme_fen_avec_historiques_differents_a_une_autre_identite():
    """Le confondant du banc : deux ordres de developpement peuvent donner la
    meme FEN, mais pas le meme tenseur d'historique, donc pas la meme
    identite reseau."""
    # Les deux lignes finissent par la meme poussee double, sinon toFEN les
    # distingue par sa case en passant, renseignee a chaque poussee double.
    plateau_a = _plateau(["g1f3", "e7e5", "e2e4", "d7d5"])
    plateau_b = _plateau(["e2e4", "e7e5", "g1f3", "d7d5"])

    assert plateau_a.to_fen() == plateau_b.to_fen()
    tenseur_a = np.asarray(plateau_a.get_alphazero_tensor())
    tenseur_b = np.asarray(plateau_b.get_alphazero_tensor())
    assert not np.array_equal(tenseur_a, tenseur_b)
    assert position_identity(plateau_a) != position_identity(plateau_b)


def test_un_joueur_bot_est_conserve():
    par_partie = {}
    for candidat in extract_candidates(_parties(), SOURCE):
        par_partie[candidat["game_id"]] = candidat

    assert par_partie["aaaaaaaa"]["black_type"] == "bot"
    assert par_partie["aaaaaaaa"]["white_type"] == "unknown"
    assert par_partie["bbbbbbbb"]["white_type"] == "bot"
    assert par_partie["bbbbbbbb"]["black_type"] == "unknown"


# ============================================================
#                 STABILITE DE L'EXTRACTION
# ============================================================

def test_melanger_les_parties_ne_change_pas_les_candidats():
    parties = _parties()
    reference = extract_candidates(parties, SOURCE)

    random.Random(20260930).shuffle(parties)
    melange = extract_candidates(parties, SOURCE)

    assert _resume(melange) == _resume(reference)


def test_commentaires_et_variantes_sans_effet(tmp_path):
    texte = BROADCASTS.read_text(encoding="utf-8")
    sans = re.sub(r"\{[^}]*\}", "", texte)
    sans = re.sub(r"\([^)]*\)", "", sans)
    propre = tmp_path / "broadcasts_propres.pgn"
    propre.write_text(sans, encoding="utf-8")

    reference = extract_candidates(_parties(), SOURCE)
    epure = extract_candidates(list(read_games(propre, SOURCE)), SOURCE)

    assert _resume(epure) == _resume(reference)


# ============================================================
#                       TELECHARGEMENT
# ============================================================

URL = ("https://database.lichess.org/broadcasts/"
       "lichess-broadcasts-2026-08.pgn.zst")


def _fetch_par_blocs(contenu):
    def fetch(_url):
        yield contenu[:50]
        yield contenu[50:]
    return fetch


def test_download_publie_apres_controle(tmp_path):
    contenu = b"archive factice" * 100
    digest = hashlib.sha256(contenu).hexdigest()
    chemin = tmp_path / "archives" / "lichess-broadcasts-2026-08.pgn.zst"

    source = download_archive(URL, chemin, digest,
                              fetch=_fetch_par_blocs(contenu))

    assert chemin.read_bytes() == contenu
    assert not chemin.with_name(chemin.name + ".part").exists()
    assert source["archive_sha256"] == digest
    assert source["month"] == "2026-08"
    assert source["source_id"] == "lichess-broadcasts-2026-08"
    assert source["license"] == "CC BY-SA 4.0"
    assert source["attribution"] == "Lichess"


def test_download_refuse_un_hash_incorrect_sans_rien_publier(tmp_path):
    contenu = b"archive factice" * 100
    chemin = tmp_path / "broadcasts-2026-08.pgn.zst"

    with pytest.raises(ValueError):
        download_archive(URL, chemin, "0" * 64,
                         fetch=_fetch_par_blocs(contenu))

    assert not chemin.exists()
    assert not chemin.with_name(chemin.name + ".part").exists()


def test_dedup_departage_les_cles_canoniques_egales():
    """Des parties qui partagent l'identifiant de diffusion donnent la meme
    cle canonique ; le position_id doit departager sans comparer les dicts."""

    def candidat(position_id, game_id, ply):
        return {"position_id": position_id, "game_id": game_id, "ply": ply,
                "source_id": "s"}

    a = cast(Position, candidat("a" * 64, "meme-diffusion", 10))
    b = cast(Position, candidat("b" * 64, "meme-diffusion", 10))

    assert [c["position_id"] for c in dedupliquer_candidats([a, b])] == [
        "a" * 64, "b" * 64]
    assert [c["position_id"] for c in dedupliquer_candidats([b, a])] == [
        "a" * 64, "b" * 64]


def test_extraction_exclut_la_troisieme_repetition(monkeypatch):
    board = chess.Board()
    coups = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + [
        "e2e4", "e7e5", "b1c3", "b8c6", "d2d4", "e5d4",
        "f1c4", "f8c5",
    ]
    for coup in coups:
        board.push_uci(coup)
    partie = chess.pgn.Game.from_board(board)
    partie.headers.update({"Event": "Repetition", "White": "A",
                           "Black": "B", "WhiteElo": "2200",
                           "BlackElo": "2200", "Date": "2026.08.01",
                           "Result": "1/2-1/2"})
    # La troisieme occurrence de la position initiale serait choisie au ply 8.
    monkeypatch.setattr(position_bench_sources, "_hash_phase",
                        lambda game_id, ply: "0" if ply == 8 else "1")

    candidats = extract_candidates([partie], SOURCE)

    assert candidats
    assert all(c["ply"] != 8 for c in candidats)
    assert all(c["repetition_count"] < 3 for c in candidats)


def test_formater_progression_affiche_pourcentage_debit_et_eta():
    ligne = formater_progression("aout", 15_500_000, 31_000_000,
                                 30_000_000.0, 0.5)

    assert "50.0 %" in ligne
    assert "31.0 Mo" in ligne
    assert "Mo/s" in ligne
    assert "ETA 00:00" in ligne


def test_formater_progression_sans_taille_totale():
    ligne = formater_progression("aout", 1_500_000, None, 1_000_000.0, None)

    assert "1.5 Mo" in ligne
    assert "ETA" not in ligne


def test_barre_progression_ecrit_le_fichier_de_suivi(tmp_path):
    import io

    fichier = tmp_path / "aout.progress"
    suivi = barre_progression("aout", intervalle_s=0.0, flot=io.StringIO(),
                              fichier=fichier)

    suivi(5_000_000, 10_000_000, 2_500_000.0, 2.0)

    assert "50.0 %" in fichier.read_text(encoding="utf-8")


def test_download_interrompu_ne_publie_pas_de_fichier_partiel(tmp_path):
    def fetch_ko(_url):
        yield b"debut de telechargement"
        raise ConnectionError("coupure")

    chemin = tmp_path / "broadcasts-2026-08.pgn.zst"

    with pytest.raises(ConnectionError):
        download_archive(URL, chemin, None, fetch=fetch_ko)

    assert not chemin.exists()
    assert not chemin.with_name(chemin.name + ".part").exists()

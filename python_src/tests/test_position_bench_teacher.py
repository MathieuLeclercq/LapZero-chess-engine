"""Tests de l'annotation Stockfish du banc de positions.

Un moteur UCI factice enregistre les appels et rend des scores forces du point
de vue des blancs : le point de vue du trait et l'assemblage des etiquettes
sont donc verifiables sans binaire ni reseau. Le smoke test reel est opt-in
par POSITION_BENCH_STOCKFISH.
"""
import hashlib
import os
import sys
from pathlib import Path
from typing import cast

import chess
import chess.engine
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine
from evaluation.bench_metrics import uci_to_index
from evaluation.position_bench_metrics import (
    AnnotatedPosition,
    validate_position,
    wdl_score,
)
from evaluation.position_bench_sources import replay_position
from evaluation.position_bench_teacher import (
    NODES_ANNOTATION,
    NODES_CRIBLAGE,
    OPTIONS_REPRODUCTIBLES,
    AnnotationStore,
    annotate_move,
    annotate_position,
    audit_labels,
    cle_coup,
    cle_criblage,
    configurer_moteur,
    identifier_stockfish,
    screen_position,
)


def _option(nom, type_, defaut):
    return chess.engine.Option(nom, type_, defaut, None, None, None)


class FauxMoteur:
    """Moteur factice : scores forces du point de vue des blancs."""

    def __init__(self, cp_blancs=80, wdl_blancs=(600, 300, 100),
                 mate_blancs=None, profondeur=20, borne=False,
                 sans_wdl=False, wdl_invalide=False, pv_incoherente=False):
        self.cp_blancs = cp_blancs
        self.wdl_blancs = wdl_blancs
        self.mate_blancs = mate_blancs
        self.profondeur = profondeur
        self.borne = borne
        self.sans_wdl = sans_wdl
        self.wdl_invalide = wdl_invalide
        self.pv_incoherente = pv_incoherente
        self.id = {"name": "Faux Stockfish 1.0", "author": "tests"}
        self.options = {
            "Threads": _option("Threads", "spin", 1),
            "Hash": _option("Hash", "spin", 16),
            "UCI_LimitStrength": _option("UCI_LimitStrength", "check", False),
            "UCI_ShowWDL": _option("UCI_ShowWDL", "check", False),
            "SyzygyPath": _option("SyzygyPath", "string", ""),
            "MultiPV": _option("MultiPV", "spin", 1),
            "Ponder": _option("Ponder", "check", False),
            "Clear Hash": _option("Clear Hash", "button", None),
        }
        self.configures = []
        self.analyses = []
        self.protocol = FauxProtocole()

    def configure(self, options):
        self.configures.append(dict(options))
        for nom in options:
            if nom not in self.options:
                raise chess.engine.EngineError(f"option inconnue {nom}")

    def analysis(self, board, limit, *, multipv=None, game=None, info=None,
                 root_moves=None, options=None):
        """Comme Stockfish : une ligne exacte, puis une ligne bornee finale.

        Une recherche coupee a la limite de noeuds finit sur une iteration
        incomplete, donc bornee. Le faux moteur reproduit cette sequence pour
        que l'annotateur conserve bien la derniere ligne exacte.
        """
        coups = list(root_moves) if root_moves is not None else []
        premier = coups[0] if coups else next(iter(board.legal_moves))
        self.analyses.append({
            "board": board.copy(),
            "nodes": getattr(limit, "nodes", None),
            "root_moves": None if root_moves is None else coups,
            "game": game,
        })
        exacte = self._info(premier, board, limit)
        if self.borne:
            return _FausseAnalyse([dict(exacte, upperbound=True)])
        bornee = dict(exacte)
        bornee["depth"] = exacte["depth"] + 1
        bornee["upperbound"] = True
        return _FausseAnalyse([exacte, bornee])

    def _info(self, premier, board, limit):
        if self.mate_blancs is not None:
            score = chess.engine.PovScore(
                chess.engine.Mate(self.mate_blancs), chess.WHITE)
        else:
            score = chess.engine.PovScore(
                chess.engine.Cp(self.cp_blancs), chess.WHITE)
        resultat = {
            "score": score,
            "pv": [premier],
            "depth": self.profondeur,
            "nodes": getattr(limit, "nodes", 0) or 0,
        }
        if not self.sans_wdl:
            if self.wdl_invalide:
                resultat["wdl"] = chess.engine.PovWdl(
                    chess.engine.Wdl(600, 300, 99), chess.WHITE)
            else:
                resultat["wdl"] = chess.engine.PovWdl(
                    chess.engine.Wdl(*self.wdl_blancs), chess.WHITE)
        if self.pv_incoherente:
            resultat["pv"] = [next(iter(board.legal_moves))]
        return resultat

    def ping(self):
        return None


class _FausseAnalyse:
    """Contexte d'analyse factice, iterable comme celui de python-chess."""

    def __init__(self, infos):
        self._infos = infos

    def __enter__(self):
        return iter(self._infos)

    def __exit__(self, *exception):
        return False


class FauxProtocole:
    """Protocole brut factice : l'export ecrit un fichier factice."""

    def __init__(self):
        self.lignes = []

    def send_line(self, ligne):
        self.lignes.append(ligne)
        chemin = Path(ligne.split(" ", 1)[1])
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_bytes(b"reseau exporte")


def _position(moves_uci):
    return {"start_fen": chess.STARTING_FEN, "moves_uci": list(moves_uci)}


def _position_complete(moves_uci):
    """Position au contrat complet, coups legaux compris."""
    board = replay_position(_position(moves_uci))
    return {
        "position_id": "p" * 64, "game_id": "g",
        "game_fingerprint": "a" * 64, "source_id": "s", "event_id": "e",
        "white_id": "w", "black_id": "b",
        "white_elo": 2200, "black_elo": 2200,
        "white_type": "human", "black_type": "human",
        "game_date": "2026-08-15",
        "start_fen": chess.STARTING_FEN, "moves_uci": list(moves_uci),
        "fen": board.to_fen(), "ply": len(moves_uci),
        "turn": 1 if board.turn == chess_engine.Color.BLACK else 0,
        "halfmove_clock": board.half_move_clock, "repetition_count": 1,
        "phase": "ouverture",
        "legal_indices": sorted(board.get_legal_move_indices()),
    }


def _annotee(position_id, scores):
    labels = [{"index": index, "score": score}
              for index, score in sorted(scores.items())]
    return cast(AnnotatedPosition, {
        "position_id": position_id,
        "labels": labels,
        "s_best": max(scores.values()),
    })


# ============================================================
#                    POINT DE VUE ET ETIQUETTES
# ============================================================

def test_score_du_point_de_vue_des_noirs():
    info = {"score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
            "wdl": chess.engine.PovWdl(chess.engine.Wdl(600, 300, 100),
                                       chess.WHITE)}

    assert info["score"].pov(chess.BLACK).score() == -80
    wdl = info["wdl"].pov(chess.BLACK)
    assert (wdl.wins, wdl.draws, wdl.losses) == (100, 300, 600)
    assert wdl_score((wdl.wins, wdl.draws, wdl.losses)) == .25


def test_annotate_move_renverse_le_point_de_vue():
    moteur = FauxMoteur(cp_blancs=80, wdl_blancs=(600, 300, 100))
    position = _position(["e2e4"])  # noirs au trait

    label = annotate_move(moteur, position, "e7e5", 1234)

    assert label["cp"] == -80
    assert label["mate"] is None
    assert label["wdl"] == (100, 300, 600)
    assert label["score"] == pytest.approx(0.25)
    assert label["nodes"] == 1234
    assert label["depth"] == 20
    assert label["index"] == uci_to_index(replay_position(position), "e7e5")
    assert moteur.analyses[-1]["root_moves"] == [chess.Move.from_uci("e7e5")]
    assert {"Clear Hash": None} in moteur.configures


def test_les_analyses_partagent_le_jeton_de_partie():
    """Un jeton stable evite le ucinewgame redondant avec le Clear Hash."""
    moteur = FauxMoteur()
    position = _position(["e2e4"])
    annotate_move(moteur, position, "e7e5", 1000)
    annotate_move(moteur, position, "g8f6", 1000)

    premier, second = moteur.analyses[0]["game"], moteur.analyses[1]["game"]
    assert premier is not None
    assert premier is second


def test_annotate_move_transmet_un_mat():
    moteur = FauxMoteur(mate_blancs=3)

    label = annotate_move(moteur, _position([]), "e2e4", 1000)

    assert label["mate"] == 3
    assert label["cp"] is None
    assert label["score"] == pytest.approx(0.75)


def test_annotate_move_refuse_un_coup_illegal():
    moteur = FauxMoteur()

    with pytest.raises(ValueError):
        annotate_move(moteur, _position(["e2e4"]), "e2e5", 1000)
    assert not moteur.analyses


@pytest.mark.parametrize("drapeau", ["sans_wdl", "borne", "wdl_invalide",
                                     "pv_incoherente"])
def test_annotation_invalide_refusee(drapeau):
    moteur = FauxMoteur(**{drapeau: True})

    with pytest.raises(RuntimeError):
        annotate_move(moteur, _position(["e2e4"]), "e7e5", 1000)


def test_annotate_position_couvre_tous_les_coups_legaux():
    moteur = FauxMoteur()
    position = _position_complete(["e2e4"])  # noirs au trait

    annotee = annotate_position(moteur, position, 500)

    validate_position(annotee)
    indices = sorted(replay_position(position).get_legal_move_indices())
    assert [label["index"] for label in annotee["labels"]] == indices
    assert len(moteur.analyses) == len(indices)
    assert annotee["s_best"] == pytest.approx(0.25)
    assert annotee["wdl_bucket"] == "avantage"


def test_screen_position_renvoie_l_esperance_du_trait():
    moteur = FauxMoteur()

    assert screen_position(moteur, _position([])) == pytest.approx(0.75)
    assert screen_position(moteur, _position(["e2e4"])) == pytest.approx(0.25)
    assert moteur.analyses[-1]["root_moves"] is None
    assert moteur.analyses[-1]["nodes"] == NODES_CRIBLAGE


# ============================================================
#                        AUDIT
# ============================================================

def test_audit_labels_passe_sur_des_etiquettes_stables():
    base = [_annotee("p1", {1: 0.8, 2: 0.6})]
    profonde = [_annotee("p1", {1: 0.8, 2: 0.6})]

    rapport = audit_labels(base, profonde)

    assert rapport["passed"] is True
    assert rapport["count"] == 1
    assert rapport["fraction_regret_ok"] == 1.0
    assert rapport["variation_moyenne"] == pytest.approx(0.0)


def test_audit_labels_echoue_si_le_meilleur_coup_de_base_regresse():
    base = [_annotee("p1", {1: 0.8, 2: 0.6})]
    profonde = [_annotee("p1", {1: 0.7, 2: 0.8})]

    rapport = audit_labels(base, profonde)

    assert rapport["passed"] is False
    assert rapport["fraction_regret_ok"] == 0.0
    assert rapport["variation_moyenne"] == pytest.approx(0.0)


def test_audit_labels_departage_par_le_plus_petit_index():
    base = [_annotee("p1", {5: 0.8, 2: 0.8})]
    profonde = [_annotee("p1", {5: 0.6, 2: 0.8})]

    rapport = audit_labels(base, profonde)

    # Le meilleur de base est le coup 2, pas le 5 : le regret est nul.
    assert rapport["passed"] is True
    assert rapport["fraction_regret_ok"] == 1.0


def test_audit_labels_compte_la_variation_en_valeur_absolue():
    base = [_annotee("p1", {1: 0.8}), _annotee("p2", {1: 0.2})]
    profonde = [_annotee("p1", {1: 0.805}), _annotee("p2", {1: 0.195})]

    rapport = audit_labels(base, profonde)

    assert rapport["variation_moyenne"] == pytest.approx(0.005)
    assert rapport["passed"] is True


def test_audit_labels_refuse_une_position_absente():
    with pytest.raises(ValueError):
        audit_labels([_annotee("p1", {1: 0.5})], [])


# ============================================================
#                  CONFIGURATION ET IDENTIFICATION
# ============================================================

def test_configurer_moteur_fixe_les_options_de_reproductibilite():
    moteur = FauxMoteur()

    configurer_moteur(moteur)

    assert moteur.configures[-1] == OPTIONS_REPRODUCTIBLES


def test_configurer_moteur_refuse_une_option_absente():
    moteur = FauxMoteur()
    del moteur.options["UCI_ShowWDL"]

    with pytest.raises(RuntimeError):
        configurer_moteur(moteur)


def test_identifier_stockfish_hache_le_binaire_et_le_reseau_externe(tmp_path):
    binaire = tmp_path / "stockfish.exe"
    binaire.write_bytes(b"binaire")
    reseau = tmp_path / "nn-abc.nnue"
    reseau.write_bytes(b"reseau")
    moteur = FauxMoteur()
    moteur.options["EvalFile"] = _option("EvalFile", "string", str(reseau))

    identite = identifier_stockfish(moteur, binaire)

    assert identite["name"] == "Faux Stockfish 1.0"
    assert identite["binary_sha256"] == hashlib.sha256(b"binaire").hexdigest()
    assert identite["networks"] == [{
        "option": "EvalFile",
        "name": str(reseau),
        "sha256": hashlib.sha256(b"reseau").hexdigest(),
        "origin": "fichier",
    }]


def test_identifier_stockfish_exporte_un_reseau_embarque(tmp_path):
    binaire = tmp_path / "stockfish.exe"
    binaire.write_bytes(b"binaire")
    moteur = FauxMoteur()
    moteur.options["EvalFile"] = _option("EvalFile", "string",
                                         "nn-embedded.nnue")

    identite = identifier_stockfish(moteur, binaire,
                                    dossier_export=tmp_path / "export")

    assert identite["networks"][0]["origin"] == "export"
    assert identite["networks"][0]["name"] == "nn-embedded.nnue"
    assert identite["networks"][0]["sha256"] == hashlib.sha256(
        b"reseau exporte").hexdigest()
    assert moteur.protocol.lignes[0].startswith("export_net ")


def test_identifier_stockfish_refuse_un_reseau_non_identifiable(tmp_path):
    binaire = tmp_path / "stockfish.exe"
    binaire.write_bytes(b"binaire")
    moteur = FauxMoteur()  # aucune option EvalFile

    with pytest.raises(RuntimeError):
        identifier_stockfish(moteur, binaire)


def test_identifier_stockfish_refuse_un_reseau_embarque_sans_export(tmp_path):
    binaire = tmp_path / "stockfish.exe"
    binaire.write_bytes(b"binaire")
    moteur = FauxMoteur()
    moteur.options["EvalFile"] = _option("EvalFile", "string",
                                         "nn-embedded.nnue")

    with pytest.raises(RuntimeError):
        identifier_stockfish(moteur, binaire)


# ============================================================
#                      CACHE DE TRAVAIL
# ============================================================

CONFIG_A = {"engine": {"name": "SF", "binary_sha256": "a" * 64},
            "nodes": NODES_ANNOTATION}
CONFIG_B = dict(CONFIG_A, nodes=2 * NODES_ANNOTATION)


def test_store_reprend_apres_interruption(tmp_path):
    store = AnnotationStore(tmp_path, CONFIG_A)
    store.put("move|p1|e2e4|200000", {"uci": "e2e4", "score": 0.5})
    store.close()

    repris = AnnotationStore(tmp_path, CONFIG_A)

    assert repris.get("move|p1|e2e4|200000") == {"uci": "e2e4", "score": 0.5}
    assert repris.get("move|p1|d2d4|200000") is None
    repris.close()


def test_store_refuse_une_autre_configuration(tmp_path):
    AnnotationStore(tmp_path, CONFIG_A).close()

    with pytest.raises(ValueError):
        AnnotationStore(tmp_path, CONFIG_B)


def test_les_cles_separent_les_budgets_et_les_positions():
    position = {"position_id": "p" * 64}
    autre = {"position_id": "q" * 64}

    assert cle_coup(position, "e2e4", NODES_ANNOTATION) != cle_coup(
        position, "e2e4", 2 * NODES_ANNOTATION)
    assert cle_coup(position, "e2e4", NODES_ANNOTATION) != cle_coup(
        autre, "e2e4", NODES_ANNOTATION)
    assert cle_criblage(position).endswith(str(NODES_CRIBLAGE))


# ============================================================
#                    SMOKE TEST REEL (OPT-IN)
# ============================================================

@pytest.mark.skipif(not os.environ.get("POSITION_BENCH_STOCKFISH"),
                    reason="POSITION_BENCH_STOCKFISH non defini")
def test_smoke_reel_stockfish(tmp_path):
    chemin = os.environ["POSITION_BENCH_STOCKFISH"]
    moteur = chess.engine.SimpleEngine.popen_uci(chemin)
    try:
        configurer_moteur(moteur)
        identite = identifier_stockfish(moteur, chemin,
                                        dossier_export=tmp_path / "nnue")
        label = annotate_move(moteur, _position(["e2e4"]), "e7e5", 5_000)

        assert identite["name"]
        assert label["nodes"] >= 1
        assert 0.0 <= label["score"] <= 1.0
    finally:
        moteur.quit()

import csv
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

from dev_tools.tt_policy_comparison import compare_policy_csv, format_report


CHAMPS = ("ligne", "coup_recherche", "reussi_recherche", "duree_s", "erreur")


def _csv(path: Path, rows: list[tuple]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CHAMPS)
        writer.writerows(rows)
    return path


def test_compare_les_memes_puzzles_ligne_par_ligne(tmp_path):
    legacy = _csv(tmp_path / "legacy.csv", [
        (0, "e2e4", True, 1.0, ""),
        (1, "d2d4", False, 2.0, ""),
        (2, "g1f3", False, 3.0, ""),
        (3, "c2c4", False, 4.0, ""),
        (4, "", False, 9.0, "solution_illegale"),
    ])
    h0 = _csv(tmp_path / "h0.csv", [
        (0, "e2e4", True, 1.5, ""),
        (1, "g1f3", True, 2.5, ""),
        (2, "g1f3", True, 3.5, ""),
        (3, "e2e4", True, 4.5, ""),
        (4, "a2a3", True, 9.0, ""),
        (5, "a2a3", True, 8.0, ""),
    ])

    comparaison = compare_policy_csv({"legacy": legacy, "h0": h0})

    assert comparaison.common_lines == 4
    assert [(p.name, p.total, p.solved, p.solve_rate, p.duration_s)
            for p in comparaison.policies] == [
        ("legacy", 4, 1, 0.25, 10.0),
        ("h0", 4, 4, 1.0, 12.0),
    ]
    paire = comparaison.pairs[0]
    assert (paire.left, paire.right) == ("legacy", "h0")
    assert paire.left_only == 0
    assert paire.right_only == 3
    assert paire.move_agreement == pytest.approx(0.5)
    assert paire.solve_rate_delta == pytest.approx(0.75)
    assert paire.mcnemar_p == pytest.approx(0.24821307899)


def test_refuse_un_identifiant_de_ligne_duplique(tmp_path):
    duplique = _csv(tmp_path / "duplique.csv", [
        (7, "e2e4", True, 1.0, ""),
        (7, "d2d4", False, 1.0, ""),
    ])

    with pytest.raises(ValueError, match="duplique"):
        compare_policy_csv({"h0": duplique})


def test_rapport_resume_les_politiques_et_les_paires(tmp_path):
    a = _csv(tmp_path / "a.csv", [(0, "e2e4", True, 1.0, "")])
    b = _csv(tmp_path / "b.csv", [(0, "d2d4", False, 2.0, "")])

    texte = format_report(compare_policy_csv({"h0": a, "h1": b}))

    assert "h0" in texte
    assert "h1" in texte
    assert "McNemar" in texte
    assert chr(0x2014) not in texte

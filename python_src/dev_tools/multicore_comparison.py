"""Comparaison appariee de deux campagnes de puzzles.

Le critere principal est fige : premier coup de recherche egal au premier coup
solution, colonne reussi_recherche. La comparaison appariee porte sur les memes
lignes, identifiees par l'index de ligne du fichier de banc. Aucune
intersection silencieuse : une ligne manquante, un doublon, une ligne en erreur
ou un sidecar incompatible font echouer la comparaison.

Verdicts :
- borne basse de l'IC95 >= -1 point : non-inferiorite ;
- borne haute < -1 point : regression ;
- entre les deux : indetermine, sans assouplissement du seuil.

Voir docs/etudes/2026-09-19-recherche-mcts-multicoeur/protocole/2026-09-14-mcts-multicore-design.md, sections 11.2
et 12.
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

RACINE_PYTHON = _runtime.PYTHON_ROOT
if str(RACINE_PYTHON) not in sys.path:
    sys.path.insert(0, str(RACINE_PYTHON))

from evaluation.bench_metrics import mcnemar
from evaluation.puzzle_bench import chemin_sidecar

MARGE_NON_INFERIORITE = -0.01
TIRAGES_BOOTSTRAP = 20000

# Colonnes sans lesquelles une campagne n'est pas exploitable : sans la
# colonne de reussite, un fichier tronque produirait une comparaison vide au
# lieu d'une erreur explicite.
COLONNES_REQUISES = ("ligne", "erreur", "reussi_recherche")

# Toutes les cles qui doivent coincider entre reference et candidat. Seul le
# nombre de workers de recherche peut differer : c'est la variable etudiee.
CLES_COMPATIBLES = (
    "modele_sha256", "banc_sha256", "simulations", "search_seconds",
    "batch_size", "cache_history_depth", "sans_historique", "accelerateur",
    "critere",
)


def paired_interval(baseline: list[bool], candidate: list[bool],
                    seed: int = 42,
                    draws: int = TIRAGES_BOOTSTRAP) -> tuple[float, float, float]:
    """Delta moyen candidat moins reference et IC95 bootstrap apparie.

    Les tirages portent sur les paires completes : l'incertitude mesuree est
    celle de l'echantillon de puzzles pour ces executions, pas toute la
    variabilite d'ordonnancement multicœur.
    """
    import numpy as np

    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paires vides ou de tailles differentes")

    delta = (np.asarray(candidate, dtype=float)
             - np.asarray(baseline, dtype=float))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        samples[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def _booleen(valeur: str, ligne: int, colonne: str) -> bool:
    if valeur not in ("True", "False"):
        raise ValueError(
            f"ligne {ligne} : {colonne} vaut {valeur!r}, attendu True/False")
    return valeur == "True"


def charger_csv(chemin: Path) -> dict[int, dict]:
    """Renvoie {index de ligne: ligne brute}. Refuse doublons et erreurs."""
    lignes: dict[int, dict] = {}
    with open(chemin, encoding="utf-8", newline="") as f:
        for rang, ligne in enumerate(csv.DictReader(f), 1):
            for colonne in COLONNES_REQUISES:
                if colonne not in ligne:
                    raise ValueError(
                        f"{chemin} : colonne {colonne} absente (rang {rang})")
            if ligne["ligne"] is None:
                raise ValueError(
                    f"{chemin} : valeur de ligne absente (rang {rang})")
            index = int(ligne["ligne"])
            if index in lignes:
                raise ValueError(f"{chemin} : ligne {index} en doublon")
            if ligne.get("erreur"):
                raise ValueError(
                    f"{chemin} : ligne {index} en erreur ({ligne['erreur']})")
            lignes[index] = ligne
    return lignes


def charger_meta(chemin_csv: Path) -> dict:
    sidecar = chemin_sidecar(chemin_csv)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"sidecar de contexte absent : {sidecar} "
            "(le CSV seul ne permet pas de verifier la comparabilite)")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def verifier_compatibilite(meta_reference: dict, meta_candidat: dict) -> None:
    for cle in CLES_COMPATIBLES:
        if meta_reference.get(cle) != meta_candidat.get(cle):
            raise ValueError(
                f"metadonnees incompatibles : {cle} vaut "
                f"{meta_reference.get(cle)!r} cote reference et "
                f"{meta_candidat.get(cle)!r} cote candidat")


def verdict_qualite(intervalle_bas: float, intervalle_haut: float) -> str:
    if intervalle_bas >= MARGE_NON_INFERIORITE:
        return "non-inferiorite"
    if intervalle_haut < MARGE_NON_INFERIORITE:
        return "regression"
    return "indetermine"


@dataclass(frozen=True)
class Comparaison:
    total: int
    reussites_reference: int
    reussites_candidat: int
    delta_moyen: float
    intervalle_bas: float
    intervalle_haut: float
    discordantes_candidat_perd: int
    discordantes_candidat_gagne: int
    mcnemar_chi2: float
    mcnemar_p: float
    depassement_median_s: float | None
    depassement_p95_s: float | None
    verdict: str


def _percentile(valeurs: list[float], quantile: float) -> float:
    if not valeurs:
        return 0.0
    ordonnees = sorted(valeurs)
    position = (len(ordonnees) - 1) * quantile
    bas = int(position)
    haut = min(bas + 1, len(ordonnees) - 1)
    fraction = position - bas
    return ordonnees[bas] * (1.0 - fraction) + ordonnees[haut] * fraction


def comparer(chemin_reference: Path, chemin_candidat: Path,
             seed: int = 42) -> tuple[Comparaison, dict, dict]:
    meta_reference = charger_meta(chemin_reference)
    meta_candidat = charger_meta(chemin_candidat)
    verifier_compatibilite(meta_reference, meta_candidat)

    reference = charger_csv(chemin_reference)
    candidat = charger_csv(chemin_candidat)
    if set(reference) != set(candidat):
        manquantes = sorted(set(reference) ^ set(candidat))[:5]
        raise ValueError(
            "les deux CSV ne portent pas sur les memes lignes ; "
            f"un exemple de divergence : {manquantes}")

    index = sorted(reference)
    reussite_reference = [
        _booleen(reference[i]["reussi_recherche"], i, "reussi_recherche")
        for i in index]
    reussite_candidat = [
        _booleen(candidat[i]["reussi_recherche"], i, "reussi_recherche")
        for i in index]

    delta, bas, haut = paired_interval(
        reussite_reference, reussite_candidat, seed=seed)
    b = sum(1 for ref, cand in zip(reussite_reference, reussite_candidat)
            if ref and not cand)
    c = sum(1 for ref, cand in zip(reussite_reference, reussite_candidat)
            if not ref and cand)
    chi2, p = mcnemar(b, c)

    depassements = [
        float(candidat[i]["depassement_s"]) for i in index
        if candidat[i].get("depassement_s") not in (None, "")]
    depassement_median = (statistics.median(depassements)
                          if depassements else None)
    depassement_p95 = (_percentile(depassements, 0.95)
                       if depassements else None)

    comparaison = Comparaison(
        total=len(index),
        reussites_reference=sum(reussite_reference),
        reussites_candidat=sum(reussite_candidat),
        delta_moyen=delta,
        intervalle_bas=bas,
        intervalle_haut=haut,
        discordantes_candidat_perd=b,
        discordantes_candidat_gagne=c,
        mcnemar_chi2=chi2,
        mcnemar_p=p,
        depassement_median_s=depassement_median,
        depassement_p95_s=depassement_p95,
        verdict=verdict_qualite(bas, haut),
    )
    return comparaison, meta_reference, meta_candidat


def format_rapport(comparaison: Comparaison, meta_reference: dict,
                   meta_candidat: dict, chemin_reference: Path,
                   chemin_candidat: Path, seed: int) -> str:
    def taux(reussites: int) -> float:
        return 100.0 * reussites / comparaison.total if comparaison.total else 0.0

    def contexte_workers(meta: dict) -> str:
        return (f"{meta.get('search_workers', 1)} workers, "
                f"{meta.get('accelerateur', 'CPU')}, "
                f"batch {meta.get('batch_size', 0)}")

    budget = meta_reference.get("budget_label") or (
        f"{meta_reference.get('simulations')} simulations")
    texte = [
        "# Comparaison appariee multicœur",
        "",
        f"Reference : `{chemin_reference.name}` ({contexte_workers(meta_reference)})",
        f"Candidat  : `{chemin_candidat.name}` ({contexte_workers(meta_candidat)})",
        f"Modele : `{meta_reference.get('modele')}` "
        f"(sha256 {str(meta_reference.get('modele_sha256'))[:12]}), "
        f"banc sha256 {str(meta_reference.get('banc_sha256'))[:12]}",
        f"Budget : {budget}, TT h{meta_reference.get('cache_history_depth')}, "
        f"c_puct {meta_reference.get('c_puct')}, "
        f"seed bootstrap {seed}",
        f"Paires : {comparaison.total}",
        "",
        "Le critere est le premier coup de recherche, identique a la solution.",
        "L'intervalle est un bootstrap apparie sur les paires completes. Ce",
        "bootstrap mesure l'incertitude de l'echantillon de puzzles pour ces",
        "executions, pas toute la variabilite d'ordonnancement multicœur. Les",
        "paires discordantes et McNemar sont des diagnostics : une p-value",
        "superieure a 0,05 n'est pas une preuve d'equivalence.",
        "",
        "| | reussites | taux |",
        "|---|---|---|",
        f"| Reference | {comparaison.reussites_reference} "
        f"| {taux(comparaison.reussites_reference):.2f} % |",
        f"| Candidat | {comparaison.reussites_candidat} "
        f"| {taux(comparaison.reussites_candidat):.2f} % |",
        "",
        f"- Delta candidat moins reference : "
        f"{100 * comparaison.delta_moyen:+.2f} point",
        f"- IC95 bootstrap : [{100 * comparaison.intervalle_bas:+.2f} ; "
        f"{100 * comparaison.intervalle_haut:+.2f}] point",
        f"- Reference bonne, candidat mauvais : "
        f"{comparaison.discordantes_candidat_perd}",
        f"- Reference mauvaise, candidat bon : "
        f"{comparaison.discordantes_candidat_gagne}",
        f"- McNemar : chi2 = {comparaison.mcnemar_chi2:.2f}, "
        f"p = {comparaison.mcnemar_p:.2e}",
    ]
    if comparaison.depassement_median_s is not None:
        texte += [
            f"- Depassement du dernier appel : mediane "
            f"{1000 * comparaison.depassement_median_s:.1f} ms, p95 "
            f"{1000 * comparaison.depassement_p95_s:.1f} ms",
        ]
    texte += [
        "",
        "## Verdict",
        "",
    ]
    if comparaison.verdict == "non-inferiorite":
        texte += [
            f"**Non-inferiorite** : la borne basse de l'IC95 "
            f"({100 * comparaison.intervalle_bas:+.2f} point) ne passe pas "
            f"sous la marge de {100 * MARGE_NON_INFERIORITE:+.0f} point.",
        ]
    elif comparaison.verdict == "regression":
        texte += [
            f"**Regression** : la borne haute de l'IC95 "
            f"({100 * comparaison.intervalle_haut:+.2f} point) passe sous la "
            f"marge de {100 * MARGE_NON_INFERIORITE:+.0f} point. Arreter la "
            "campagne.",
        ]
    else:
        texte += [
            "**Indetermine** : l'IC95 chevauche la marge. Ni activation par "
            "defaut, ni assouplissement du seuil ; la campagne complete doit "
            "trancher.",
        ]
    return "\n".join(texte) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comparaison, meta_reference, meta_candidat = comparer(
        args.baseline, args.candidate, args.seed)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        format_rapport(comparaison, meta_reference, meta_candidat,
                       args.baseline, args.candidate, args.seed),
        encoding="utf-8")
    print(json.dumps(asdict(comparaison), indent=2, sort_keys=True))
    print(f"\nRapport : {args.out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

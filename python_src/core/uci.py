
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import sys
import threading
import time
import math
from pathlib import Path
from core.data_paths import ONNX_DIR, LOGS_DIR

import chess_engine
from core.lib import parse_uci_to_coords, coords_to_uci, decode_move_index, encode_move

# ============================================================
#                     CONFIGURATION EN DUR
# ============================================================
MODEL_PATH = str(
    ONNX_DIR
    / "2026_04_30_09h53_iter436_unsupervised.onnx")
DEFAULT_SIMULATIONS = 1200
# Pas de simulations entre deux controles d'horloge. Il plafonne aussi le lot
# MCTS : un lot ne peut pas collecter plus de feuilles que le palier n'en
# demande. A 400 simulations par seconde, 64 simulations valent 160 ms, ce qui
# reste plus fin que SNAPSHOT_INTERVAL.
BATCH_SIZE = 64

# Taille de lot passee au MCTS, qui evalue plusieurs positions par inference
# grace au virtual loss. Le banc de puzzles valide 8, le meilleur compromis
# entre debit et qualite sur les trois positions mesurees.
MCTS_BATCH_SIZE = 8

# Workers CPU internes au C++ par recherche, mesures par la campagne
# multicœur du 2026-09-19 : mediane +8 a +26 % selon la position, p95 de
# latence sous +5 %, qualite non-inferieure sur 2500 puzzles. Voir
# docs/etudes/2026-09-19-recherche-mcts-multicoeur/resultats/historique/2026-09-16-multicore-waves-results.md.
MCTS_WORKER_COUNT = 8

# Forme de lot fixe : chaque vague est paddee a MCTS_BATCH_SIZE avant l'appel
# ONNX. Sur le banc du 2026-09-19, le changement de forme quadruplait le cout
# de session->Run : x2.0 en ouverture, x3.1 en milieu, x2.5 en finale, et le
# p95 de latence divise par 2 a 3. Les sorties des positions dupliquees sont
# ignorees, donc la recherche est inchangee. Voir
# docs/etudes/2026-09-19-cout-calcul-cpu-gpu/protocole/2026-09-19-cout-calcul-cpu-gpu.md.
MCTS_FIXED_BATCH = True

# Virtual loss 2 : +9 a +20 % de debit mesures sur le banc de recherche, et
# qualite non-inferieure sur 2500 puzzles, IC95 [-0.4 ; +0.56] point.
# Voir docs/etudes/2026-09-19-recherche-mcts-multicoeur/resultats/historique/2026-09-22-rebalayage-virtual-loss-resultats.md.
MCTS_VIRTUAL_LOSS = 2
SNAPSHOT_INTERVAL = 0.1
NB_FAST_PLIES_OPENING = 10

# Journal par coup, une ligne par recherche. Independant du log de lichess-bot,
# qui envoie stderr vers DEVNULL quand silence_stderr est faux. Le volume est
# d'environ 100 octets par coup, et la taille est plafonnee par rotation.
STATS_LOG = LOGS_DIR / "uci_stats.log"
STATS_LOG_MAX_BYTES = 1_000_000


def _journaliser(message: str) -> None:
    """Ecrit le bilan sur stderr (terminal, Nibbler) et dans le journal."""
    print(message, file=sys.stderr, flush=True)
    try:
        STATS_LOG.parent.mkdir(parents=True, exist_ok=True)
        if (STATS_LOG.exists()
                and STATS_LOG.stat().st_size > STATS_LOG_MAX_BYTES):
            STATS_LOG.replace(STATS_LOG.with_suffix(".log.1"))
        with open(STATS_LOG, "a", encoding="utf-8") as fichier:
            fichier.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def _construire_evaluateur():
    """Le GPU d'abord, le CPU en repli.

    Le batching ne rend vraiment que sur GPU : la meme recherche passe de 284 a
    415 simulations par seconde. Mais le provider CUDA peut manquer sur la
    machine qui heberge le bot, et AppendExecutionProvider_CUDA leve alors. Un
    repli silencieux vaut mieux qu'un moteur qui refuse de demarrer.
    """
    try:
        return chess_engine.ONNXEvaluator(MODEL_PATH, True), "GPU"
    except Exception as e:
        print(f"info string GPU indisponible, repli sur le CPU : {e}", flush=True)
        return chess_engine.ONNXEvaluator(MODEL_PATH, False), "CPU"


# ============================================================
#                     MOTEUR UCI
# ============================================================
class UCIEngine:
    def __init__(self, evaluator=None, mcts=None):
        self.board = chess_engine.Chessboard()
        # Injection pour les tests : sans elle, construire un UCIEngine exige un
        # modele ONNX, et MODEL_PATH pointe vers une autre machine.
        if evaluator is not None:
            self.evaluator = evaluator
            self.provider = "injecte"
        else:
            self.evaluator, self.provider = _construire_evaluateur()
        self.mcts = (mcts if mcts is not None
                     else chess_engine.MCTS(self.evaluator, tt_size=4_000_000))
        if mcts is None:
            self.mcts.set_fixed_batch(MCTS_FIXED_BATCH)
            self.mcts.set_tuning(MCTS_VIRTUAL_LOSS)
        self.search_thread = None

        self.stop_event = threading.Event()

        # Variables de contrôle de l'horloge
        self.is_pondering = False
        self.is_infinite = False
        self.target_time = 10.0
        self.search_start_time = 0.0

        # Historique pour le Root Shifting
        self.last_move_list = []
        self.real_ply = 0
        # Identite de la position de base de la derniere commande position :
        # "startpos" ou les six champs FEN. Sans elle, deux bases differentes
        # suivies de la meme liste de coups seraient traitees comme une simple
        # avancee, et la racine serait decalee sur un plateau etranger.
        self.base_identity = None

    @staticmethod
    def q_to_cp(q_value):
        q = max(-0.999999, min(0.999999, q_value))
        return int(round(290.0 * math.atanh(q)))

    def loop(self):
        while True:
            try:
                line = sys.stdin.readline().strip()
                if not line:
                    continue
            except EOFError:
                break

            tokens = line.split()
            if not tokens:
                continue
            command = tokens[0]

            if command == "uci":
                print("id name Lc0 Custom")
                print("id author Mathieu Leclercq")
                print("option name WeightsFile type string default <internal>")
                print("uciok")
                sys.stdout.flush()
                # Bilan de configuration : stderr pour le direct, journal pour
                # l'apres-coup.
                _journaliser(
                    f"uci.py pret : modele {Path(MODEL_PATH).name}, "
                    f"{self.provider}, lot {MCTS_BATCH_SIZE}, "
                    f"lot fixe {MCTS_FIXED_BATCH}, "
                    f"workers {MCTS_WORKER_COUNT}, TT 4000000")

            elif command == "isready":
                print("readyok")
                sys.stdout.flush()

            elif command == "ucinewgame":
                self.mcts.reset_analysis()
                self.last_move_list = []
                self.base_identity = None

            elif command == "position":
                self.parse_position(tokens[1:])

            elif command == "go":
                self.start_search(tokens[1:])

            elif command == "ponderhit":
                # L'adversaire a joué notre ponder move !
                # On bascule en temps normal sans arrêter la recherche C++
                self.is_pondering = False
                self.search_start_time = time.time()  # Le chrono démarre !

            elif command == "stop":
                self.stop_search()

            elif command == "quit":
                self.stop_search()
                break

    @staticmethod
    def _base_identity(tokens):
        """Identite de la position de base : "startpos" explicite, ou les six
        champs FEN normalises sur les espaces. Le trait, les droits de roque,
        la prise en passant et les compteurs en font partie : deux commandes
        qui ne different que par eux ne sont pas la meme position."""
        if len(tokens) > 0 and tokens[0] == "startpos":
            return "startpos"
        if len(tokens) > 0 and tokens[0] == "fen" and len(tokens) >= 7:
            return "fen " + " ".join(tokens[1:7])
        return None

    def parse_position(self, tokens):
        # La recherche doit etre arretee AVANT toute modification de l'arbre ou
        # du plateau. update_root detruit le reste de l'arbre par unique_ptr, et
        # le fil de recherche y descend encore. Voir
        # docs/superpowers/specs/2026-09-11-search-bench-design.md section 3.
        self.stop_search()

        moves_idx = -1
        if len(tokens) > 0 and tokens[0] == "startpos":
            moves_idx = 2 if len(tokens) > 1 and tokens[1] == "moves" else -1
        elif len(tokens) > 0 and tokens[0] == "fen":
            moves_idx = 8 if len(tokens) > 7 and tokens[7] == "moves" else -1

        new_move_list = tokens[moves_idx:] if moves_idx != -1 else []

        nouvelle_base = self._base_identity(tokens)
        meme_base = (nouvelle_base is not None
                     and nouvelle_base == self.base_identity)

        # 1. Ponder Hit Parfait ou redondance GUI (On ne touche à rien)
        if meme_base and len(self.last_move_list) > 0 and new_move_list == self.last_move_list:
            pass

        # 2. Avancée normale d'un coup (On décale la racine)
        elif (meme_base and len(self.last_move_list) > 0
              and len(new_move_list) == len(self.last_move_list) + 1
              and new_move_list[:-1] == self.last_move_list):
            last_uci = new_move_list[-1]
            is_black = (self.board.turn == chess_engine.Color.BLACK)
            orig_f, orig_r, dest_f, dest_r, promo = parse_uci_to_coords(last_uci)

            move_idx = encode_move(orig_f, orig_r, dest_f, dest_r, promo, is_black)
            self.mcts.update_root(move_idx)
            self.board.move_piece(orig_f, orig_r, dest_f, dest_r, promo)

        # 3. 1er coup de la partie, Ponder Miss, Nouvelle Partie ou base
        # différente : on reconstruit tout PROPREMENT
        else:
            self.mcts.reset_analysis()
            self.board = chess_engine.Chessboard()

            if len(tokens) > 0 and tokens[0] == "startpos":
                self.board.set_startup_pieces()
            elif len(tokens) > 0 and tokens[0] == "fen":
                fen_string = " ".join(tokens[1:7])
                self.board.load_fen(fen_string)

            for uci_move in new_move_list:
                orig_f, orig_r, dest_f, dest_r, promo = parse_uci_to_coords(uci_move)
                self.board.move_piece(orig_f, orig_r, dest_f, dest_r, promo)

        self.last_move_list = new_move_list
        self.base_identity = nouvelle_base

        if len(tokens) > 0 and tokens[0] == "startpos":
            self.real_ply = len(new_move_list)
        elif len(tokens) > 0 and tokens[0] == "fen":
            try:
                turn = tokens[2]
                fullmove = int(tokens[6])
                base_ply = (max(1, fullmove) - 1) * 2 + (1 if turn == 'b' else 0)
                self.real_ply = base_ply + len(new_move_list)
            except (ValueError, IndexError):
                self.real_ply = len(new_move_list)

    def _should_stop_early(self, history, elapsed, total_sims):
        """
        Décide si on peut jouer le coup plus vite.
        Retourne True si on doit stopper la recherche.
        """
        if total_sims < 3000:
            return False

        if len(history) < 5:
            return False

        latest = history[-1]
        best_move, best_visits, second_visits = latest[1], latest[2], latest[3]

        if second_visits == 0:
            return False

        ratio = best_visits / second_visits

        # Critère 1 : dominance écrasante + stabilité, MAIS on le force à
        # consommer au moins 25% de son temps pour être sûr de lui.
        if ratio >= 10.0 and elapsed >= self.target_time * 0.25:
            recent_bests = [h[1] for h in history[-5:]]
            if all(m == best_move for m in recent_bests):
                return True

        # Critère 2 : dominance modérée (3x) + temps déjà consommé (>= 70%) + stabilité
        if ratio >= 3.0 and elapsed >= self.target_time * 0.7:
            recent_bests = [h[1] for h in history[-3:]]
            if all(m == best_move for m in recent_bests):
                old_ratio = history[-5][2] / max(1, history[-5][3])
                if ratio >= old_ratio * 0.9:
                    return True

        return False

    @staticmethod
    def _should_extend_time(history):
        """Si la position est incertaine (best_move change souvent),
        demande du temps supplémentaire."""
        if len(history) < 10:
            return False

        changes = sum(1 for i in range(1, 10) if history[-i][1] != history[-i - 1][1])
        return changes >= 2

    def start_search(self, tokens):
        self.stop_search()
        self.stop_event.clear()

        my_time = None
        my_inc = 0
        movetime_ms = None

        # Détection de l'ordre de Pondering par la GUI
        self.is_pondering = "ponder" in tokens
        self.is_infinite = "infinite" in tokens
        is_black = (self.board.turn == chess_engine.Color.BLACK)

        for i in range(len(tokens) - 1):
            if tokens[i] == "wtime" and not is_black:
                my_time = int(tokens[i + 1])
            elif tokens[i] == "btime" and is_black:
                my_time = int(tokens[i + 1])
            elif tokens[i] == "winc" and not is_black:
                my_inc = int(tokens[i + 1])
            elif tokens[i] == "binc" and is_black:
                my_inc = int(tokens[i + 1])
            elif tokens[i] == "movetime":
                movetime_ms = int(tokens[i + 1])

        # Calcul du budget temps
        if self.is_infinite:
            self.target_time = 1e9
        elif movetime_ms is not None:
            self.target_time = (movetime_ms / 1000.0) * 0.95
        elif my_time is not None:
            # 1. Move Overhead : matelas de 100ms de survie (latence)
            safe_time = max(1, my_time - 100)

            # 2. Diviseur agressif
            if my_inc == 0:
                self.target_time = (safe_time / 25.0) / 1000.0
            else:
                self.target_time = ((safe_time / 20.0) + (my_inc * 0.85)) / 1000.0

            # 3. Limites d'urgence
            self.target_time = max(0.01, self.target_time)
            self.target_time = min(self.target_time, (safe_time / 1000.0) * 0.8)
        else:
            self.target_time = 10.0

        self.search_start_time = time.time()
        self.search_thread = threading.Thread(target=self.search_worker)
        self.search_thread.start()

    def stop_search(self):
        if self.search_thread and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join()

    def search_worker(self):
        total_sims = 0
        last_info_time = 0.0
        last_snapshot_time = 0.0
        elapsed = 0.0

        # Historique pour détecter les tendances : (elapsed, best_move, best_visits, second_visits)
        history = []

        # --- 1. BOUCLE DE RECHERCHE ---
        while not self.stop_event.is_set():

            if self.is_infinite or self.is_pondering:
                max_sims = float('inf')
            elif getattr(self, 'real_ply', 0) < NB_FAST_PLIES_OPENING:
                max_sims = DEFAULT_SIMULATIONS
            else:
                max_sims = 10_000_000

            elapsed = time.time() - self.search_start_time

            # 1a. Gestion du temps (hors ponder)
            if not self.is_pondering:
                # Temps écoulé — possibilité d'étendre jusqu'à 1.5x si position incertaine
                if elapsed >= self.target_time:
                    if self._should_extend_time(history) and elapsed < self.target_time * 1.5:
                        pass
                    else:
                        break

                # Early stop si dominance claire après 30% du temps
                if elapsed >= self.target_time * 0.3 and self._should_stop_early(
                        history, elapsed, total_sims):
                    break

            # 1b. Exécution des simulations
            if total_sims < max_sims:
                sims_to_do = min(BATCH_SIZE, max_sims - total_sims)
                self.mcts.step_analysis(self.board, sims_to_do, 1.4,
                                        MCTS_BATCH_SIZE, MCTS_WORKER_COUNT)
                total_sims += sims_to_do

                # Détermine ce qu'il faut faire avec les stats actuelles
                now = time.time()
                needs_info = now - last_info_time >= 0.5
                needs_snapshot = (not self.is_pondering and
                                  now - last_snapshot_time >= SNAPSHOT_INTERVAL)

                if needs_info or needs_snapshot:
                    stats = self.mcts.get_analysis_results()

                    # coup forcé
                    if len(stats) == 1 and not self.is_pondering and not self.is_infinite:
                        break

                    # Mise à jour de l'historique de tendance (uniquement hors ponder)
                    if needs_snapshot:
                        last_snapshot_time = now
                        if len(stats) >= 2:
                            history.append((
                                elapsed,
                                stats[0].move_idx,
                                stats[0].visits,
                                stats[1].visits
                            ))
                            if len(history) > 20:
                                history.pop(0)

                    # Affichage UCI info
                    if needs_info:
                        last_info_time = now
                        stats_sorted = sorted(stats, key=lambda s: s.visits, reverse=True)

                        real_total_nodes = sum(s.visits for s in stats_sorted)
                        if real_total_nodes == 0:
                            real_total_nodes = 1

                        for multipv_idx in range(len(stats_sorted) - 1, -1, -1):
                            move_stat = stats_sorted[multipv_idx]
                            is_black = (self.board.turn == chess_engine.Color.BLACK)
                            o_f, o_r, d_f, d_r, promo = decode_move_index(
                                self.board, move_stat.move_idx, is_black)
                            uci_str = coords_to_uci(o_f, o_r, d_f, d_r, promo)
                            cp_score = self.q_to_cp(move_stat.q_value)

                            print(
                                f"info depth 1 seldepth {total_sims} "
                                f"multipv {multipv_idx + 1} score cp "
                                f"{cp_score} nodes {real_total_nodes} pv {uci_str}")

                            n = move_stat.visits
                            p = move_stat.prior * 100.0
                            q = move_stat.q_value
                            v = move_stat.v_value
                            print(
                                f"info string {uci_str} (0 ) N: {n} (+ 0) "
                                f"(P: {p:.2f}%) (Q: {q:.5f}) (V: {v:.5f})")

                        sys.stdout.flush()

            # 1c. Si on a atteint max_sims
            else:
                if self.is_pondering:
                    time.sleep(0.01)
                else:
                    break

        # --- 2. FIN DE RECHERCHE ET NORME UCI ---
        best_stats = self.mcts.get_analysis_results()
        if not best_stats:
            _journaliser("recherche : aucune statistique, bestmove 0000")
            print("bestmove 0000")
            sys.stdout.flush()
            return

        my_best = best_stats[0]
        is_black = (self.board.turn == chess_engine.Color.BLACK)
        o_f, o_r, d_f, d_r, promo = decode_move_index(self.board, my_best.move_idx, is_black)
        my_best_uci = coords_to_uci(o_f, o_r, d_f, d_r, promo)

        # Bilan par coup : numero de coup, simulations, coup choisi, deuxieme
        # coup et visites. Permet de diagnostiquer un coup faible a posteriori
        # sans dependre du log de lichess-bot.
        visites_racine = sum(s.visits for s in best_stats)
        deuxieme = ""
        if len(best_stats) > 1:
            second = best_stats[1]
            s_f, s_r, s_df, s_dr, s_promo = decode_move_index(
                self.board, second.move_idx, is_black)
            deuxieme = (f", 2e {coords_to_uci(s_f, s_r, s_df, s_dr, s_promo)}"
                        f" ({second.visits}/{visites_racine})")
        trait = "noirs" if is_black else "blancs"
        _journaliser(
            f"recherche : coup {self.real_ply // 2 + 1} {trait}, "
            f"{total_sims} sims en "
            f"{time.time() - self.search_start_time:.1f} s, "
            f"best {my_best_uci} ({my_best.visits}/{visites_racine})"
            f"{deuxieme}, workers {MCTS_WORKER_COUNT}, {self.provider}, "
            f"{'arretee' if self.stop_event.is_set() else 'terminee'}")

        # Si la GUI a forcé l'arrêt OU si on est en analyse libre, on coupe net.
        if self.stop_event.is_set() or self.target_time == 1e9:
            print(f"bestmove {my_best_uci}")
            sys.stdout.flush()
            return

        # --- 3. MODE PARTIE (Lichess) : PRÉPARATION DU PONDERING ---
        ponder_uci = ""

        if self.board.move_piece(o_f, o_r, d_f, d_r, promo):
            self.mcts.update_root(my_best.move_idx)
            self.last_move_list.append(my_best_uci)

            opp_stats = self.mcts.get_analysis_results()
            if opp_stats:
                opp_best = opp_stats[0]
                opp_is_black = (self.board.turn == chess_engine.Color.BLACK)
                opp_o_f, opp_o_r, opp_d_f, opp_d_r, opp_promo = decode_move_index(
                    self.board, opp_best.move_idx, opp_is_black)
                ponder_uci = coords_to_uci(opp_o_f, opp_o_r, opp_d_f, opp_d_r, opp_promo)

        if ponder_uci:
            print(f"bestmove {my_best_uci} ponder {ponder_uci}")
        else:
            print(f"bestmove {my_best_uci}")

        sys.stdout.flush()


if __name__ == "__main__":
    engine = UCIEngine()
    engine.loop()

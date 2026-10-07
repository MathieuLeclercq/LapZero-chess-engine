"""Sonde spawn autonome, sans entrainement ni evaluation de modele."""
import multiprocessing as mp
import sys


def probe(queue):
    import _runtime
    import chess_engine
    from evaluation.position_bench import _probabilites_masquees
    board = chess_engine.Chessboard()
    board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    queue.put((len(board.get_legal_move_indices()), "torch" in sys.modules,
               callable(_probabilites_masquees)))


if __name__ == "__main__":
    context = mp.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=probe, args=(queue,))
    child.start()
    try:
        result = queue.get(timeout=15)
        child.join(10)
        assert result == (20, False, True), result
        assert child.exitcode == 0, child.exitcode
        print("SPAWN_OK")
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        queue.close()
        queue.join_thread()

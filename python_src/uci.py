"""Lanceur compatible, implementation dans core.uci."""
import _runtime

if __name__ == "__main__":
    from core.uci import UCIEngine
    UCIEngine().loop()

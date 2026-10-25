"""Standalone executable for nonneural same-pair comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.comparison_baselines import main
if __name__ == "__main__": main()

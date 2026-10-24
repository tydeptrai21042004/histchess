#!/usr/bin/env python3
"""Run the paper-grounded causal HistChess architecture benchmark.

Example:
 python scripts/run_architecture_benchmark.py --out /kaggle/working/histchess_arch_results \
   --pairs-per-task 300 --architectures elman_rnn lstm gru causal_transformer \
   linear_attention diagonal_ssm --folds 0 1 2 --seeds 3101 3102 3103 \
   --cutoffs 0 4 8 16 full --device cpu
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.architecture_benchmark import main
if __name__ == "__main__":main()

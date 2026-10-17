#!/usr/bin/env python3
"""OPTIONAL PGN parsing utility; benchmark default needs no downloads."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chess.pgn
from histchess.core import write_jsonl
p=argparse.ArgumentParser();p.add_argument('pgn');p.add_argument('--out',default='data/pgn_positions.jsonl');p.add_argument('--max-games',type=int,default=100);a=p.parse_args()
rows=[]
with open(a.pgn,encoding='utf8',errors='replace') as file:
    for gi in range(a.max_games):
        game=chess.pgn.read_game(file)
        if game is None:break
        board=game.board();history=[]
        for move in game.mainline_moves():
            if move not in board.legal_moves: raise ValueError(f'Game {gi}: illegal move {move}')
            board.push(move);history.append(move.uci())
            rows.append({'game_index':gi,'ply':len(history),'history_uci':list(history),'fen':board.fen(en_passant='legal')})
write_jsonl(a.out,rows)
print(json.dumps({'positions':len(rows),'out':a.out}))

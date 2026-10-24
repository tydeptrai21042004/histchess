"""OPTIONAL synthetic neutral-history stress test for the v3 counterfactual pairs.

Not a corpus of realistic chess games. For castling, a common reversible
knight excursion follows the event that removed the right; for other tasks the
excursion precedes the decisive suffix (so long input != long dependency).
Every completed history is re-played with python-chess and compared against
its original hidden-state label by core.pair_from_moves.

The generator rejects sequences reaching an automatic game outcome, including
fivefold repetition and the 75-move rule. Failure to produce the requested
length is fatal: it never silently keeps a shorter pair.
"""
from __future__ import annotations

import math
import random

import chess

from .core import replay, pair_from_moves, board_key


def _automatic_outcome(board):
    return board.outcome(claim_draw=False) is not None


def _tour(start_a, start_b, moves_required, rng):
    """A synchronized, reversible, knight-only walk of 4n plies.

    Both positions must have identical layout and turn. Knight-only moves are
    never captures, promotions, or pawn moves. A forward path is inverted in
    side-to-move order; all moves and automatic game termination are checked.
    """
    if moves_required < 0 or moves_required % 4:
        raise ValueError("Tour length must be a nonnegative multiple of 4")
    if board_key(start_a) != board_key(start_b):
        raise ValueError("Tour requires identical observable positions")
    if not moves_required:
        return ()
    left=start_a.copy(stack=True);right=start_b.copy(stack=True)
    start_key=(left.board_fen(), left.turn)
    forward=[]
    for _ in range(moves_required//2):
        available=[]
        for mv in left.legal_moves:
            if (left.piece_type_at(mv.from_square)==chess.KNIGHT
                    and mv in right.legal_moves and not left.is_capture(mv)
                    and not right.is_capture(mv)):
                available.append(mv)
        if not available:raise ValueError("No reversible knight move available")
        rng.shuffle(available)
        chosen=None
        for mv in available:
            left.push(mv);right.push(mv)
            if not _automatic_outcome(left) and not _automatic_outcome(right):
                chosen=mv;break
            left.pop();right.pop()
        if chosen is None:raise ValueError("Forward tour would end the game")
        forward.append(chosen)
    # Undo each two-ply white/black pair, preserving the turn order.
    inverse=[]
    for ix in range(len(forward)-2,-1,-2):
        for mv in (forward[ix],forward[ix+1]):
            inv=chess.Move(mv.to_square,mv.from_square)
            if (inv not in left.legal_moves or inv not in right.legal_moves
                    or left.is_capture(inv) or right.is_capture(inv)):
                raise ValueError("Inverse knight move is not legal")
            left.push(inv);right.push(inv)
            if _automatic_outcome(left) or _automatic_outcome(right):
                raise ValueError("Inverse tour would end the game")
            inverse.append(inv)
    if (left.board_fen(),left.turn)!=start_key or (right.board_fen(),right.turn)!=start_key:
        raise ValueError("Neutral tour failed to restore original board")
    return tuple(m.uci() for m in forward+inverse)


def extend_pair(pair, target_plies=64, seed=3101, attempts=100):
    if target_plies < 0 or attempts < 1:raise ValueError("Invalid stress settings")
    task=pair["task"]
    pos=tuple(pair["positive"]["history_uci"])
    neg=tuple(pair["negative"]["history_uci"])
    if len(pos)!=len(neg):raise ValueError("Unequal counterfactual lengths")
    # In repetition, introduce two different irreversible pawn changes to
    # ensure that the negative example's final state is encountered only twice.
    # Neither e/d pawn move blocks the existing knight-template landing squares.
    pre=("e2e3","e7e6") if task=="repetition" else ()
    post=("d2d3","d7d6") if task=="repetition" else ()
    requested_extra=max(0,target_plies-len(pos)-len(pre)-len(post))
    extra=int(math.ceil(requested_extra/4))*4
    rng=random.Random(f"{seed}|{pair['pair_id']}|{target_plies}")
    failures=[]
    for _ in range(attempts):
        try:
            if task=="castling":
                a,b=replay(pos),replay(neg)
                added=_tour(a,b,extra,rng)
                new_pos=pos+added;new_neg=neg+added
            else:
                start_a=replay(pre); start_b=replay(pre)
                added=_tour(start_a,start_b,extra,rng)
                prefix=pre+added+post
                new_pos=prefix+pos;new_neg=prefix+neg
            rebuilt=pair_from_moves(task,new_pos,new_neg)
            # Check all intermediate states, not only the final certificate.
            for history in (new_pos,new_neg):
                board=chess.Board()
                for uci in history:
                    board.push_uci(uci)
                    if _automatic_outcome(board):
                        raise ValueError("Intermediate automatic game outcome")
            if len(new_pos)<target_plies:
                raise ValueError("Requested history minimum was not achieved")
            rebuilt["generator_family"]=pair["generator_family"]
            rebuilt["split"]=pair.get("split","unassigned")
            rebuilt["stress_protocol"]="synthetic_reversible_knight_tour_v1"
            rebuilt["first_divergence_distance"]=(len(new_pos)-next(
                (i for i,(x,y) in enumerate(zip(new_pos,new_neg)) if x!=y),len(new_pos))-1)
            return rebuilt
        except (ValueError,AssertionError) as exc:
            failures.append(str(exc))
    raise RuntimeError(f"Cannot create a certified {target_plies}-ply {task} pair "
                       f"after {attempts} attempts; last failure: {failures[-1] if failures else 'unknown'}")


def extend_pairs(pairs,target_plies=64,seed=3101,attempts=100):
    result=[]
    used=set()
    for ix,p in enumerate(pairs):
        q=extend_pair(p,target_plies,seed,attempts)
        if q["board_key"] in used:
            raise RuntimeError(f"Stress construction duplicated the observable board at pair {ix}")
        used.add(q["board_key"])
        result.append(q)
    return result

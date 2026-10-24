"""Certified synthetic long-history stress data for HistChess.

This is NOT a real-game benchmark.  Castling rights have a long causal delay:
we append a common legal continuation after the divergent event.  For the
other rule tasks, a synthetic common prefix precedes the decisive task suffix,
so input length must not be conflated with causal-memory distance.

All returned histories are checked by ``pair_from_moves`` and replayed move by
move against python-chess, including automatic game outcomes.  No failures are
silently skipped or replaced with shorter inputs.
"""
from __future__ import annotations

import random
import chess

from .core import board_key, observed_label, pair_from_moves, replay


def _automatic_outcome(board):
    return board.outcome(claim_draw=False) is not None


def _verify_trajectory(history):
    board=chess.Board()
    for ply, uci in enumerate(history, 1):
        move=chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {ply}: {uci}")
        board.push(move)
        if _automatic_outcome(board):
            raise ValueError(f"Automatic game outcome at ply {ply}")
    return board


def _common_castling_suffix(pos, neg, missing, rng):
    """Quiet, shared continuation, keeping WHITE kingside rights different.

    Unlike a reversible knight-only tour, this works even when there are no
    knight moves and uses irreversible pawn advances to avoid fivefold draws.
    It does not require the board to return to the initial layout: both sides
    of the *counterfactual pair* receive exactly the same continuation.
    """
    a,b=replay(pos),replay(neg)
    if board_key(a)!=board_key(b):
        raise ValueError("Castling starting layouts differ")
    if not (observed_label(a,"castling") and not observed_label(b,"castling")):
        raise ValueError("Starting castling labels are invalid")
    suffix=[]
    for _ in range(missing):
        alternatives=[]
        # Shared legal moves are essential: rights can affect legal castling.
        right_legal=set(b.legal_moves)
        for move in a.legal_moves:
            if move not in right_legal or a.is_capture(move) or b.is_capture(move):
                continue
            if move.promotion or a.is_castling(move) or b.is_castling(move):
                continue
            piece=a.piece_type_at(move.from_square)
            if piece == chess.KING:
                continue
            # Avoid forcing king/rook rights changes, even on the negative side.
            if piece == chess.ROOK and a.turn == chess.WHITE:
                continue
            a.push(move);b.push(move)
            valid=(board_key(a)==board_key(b)
                   and observed_label(a,"castling")
                   and not observed_label(b,"castling")
                   and not _automatic_outcome(a) and not _automatic_outcome(b))
            a.pop();b.pop()
            if valid:
                # Pawn pushes and simple piece moves are less likely to trap
                # the random continuation than queen/rook excursions.
                priority={chess.PAWN:5,chess.KNIGHT:5,chess.BISHOP:3,
                          chess.QUEEN:1,chess.ROOK:1}.get(piece,1)
                alternatives.extend([move]*priority)
        if not alternatives:
            raise ValueError(f"No safe shared castling continuation at {len(suffix)} plies")
        move=rng.choice(alternatives)
        a.push(move);b.push(move)
        suffix.append(move.uci())
    return tuple(suffix)


def _pawn_start_squares_used_by_suffix(pos, neg):
    """Reserve pawns that the decisive suffix may need at their start square."""
    reserved=set()
    start=chess.Board()
    for uci in pos+neg:
        source=uci[:2]
        sq=chess.parse_square(source)
        if start.piece_type_at(sq)==chess.PAWN:
            reserved.add(sq)
    return reserved


def _quiet_pawn_pair(board, reserved, pos, neg, prefix, rng):
    """A two-ply irreversible block that leaves both target suffixes legal."""
    candidates=[]
    for white in list(board.legal_moves):
        if (board.piece_type_at(white.from_square)!=chess.PAWN
                or white.from_square in reserved or board.is_capture(white)
                or white.promotion):
            continue
        board.push(white)
        if not _automatic_outcome(board):
            for black in list(board.legal_moves):
                if (board.piece_type_at(black.from_square)!=chess.PAWN
                        or black.from_square in reserved or board.is_capture(black)
                        or black.promotion):
                    continue
                board.push(black)
                if not _automatic_outcome(board):
                    trial=prefix+(white.uci(),black.uci())
                    try:
                        # The suffix can depend on empty knight destinations,
                        # specific unmoved pawns, en-passant and repetition.
                        # Checking both sides before committing is necessary.
                        replay(trial+pos)
                        replay(trial+neg)
                        candidates.append((white,black))
                    except ValueError:
                        pass
                board.pop()
        board.pop()
    if not candidates:
        raise ValueError("No suffix-compatible irreversible pawn block available")
    white,black=rng.choice(candidates)
    board.push(white);board.push(black)
    return (white.uci(),black.uci())


def _knight_cycle(board,rng):
    """Four legal plies: white knight, black knight, then both return."""
    choices=[]
    for wm in list(board.legal_moves):
        if board.piece_type_at(wm.from_square)!=chess.KNIGHT or board.is_capture(wm):
            continue
        board.push(wm)
        for bm in list(board.legal_moves):
            if board.piece_type_at(bm.from_square)!=chess.KNIGHT or board.is_capture(bm):
                continue
            board.push(bm)
            wi=chess.Move(wm.to_square,wm.from_square)
            if wi in board.legal_moves and not board.is_capture(wi):
                board.push(wi)
                bi=chess.Move(bm.to_square,bm.from_square)
                if bi in board.legal_moves and not board.is_capture(bi):
                    board.push(bi)
                    if not _automatic_outcome(board):
                        choices.append((wm,bm,wi,bi))
                    board.pop()
                board.pop()
            board.pop()
        board.pop()
    if not choices:
        raise ValueError("No safe four-ply knight cycle")
    selected=rng.choice(choices)
    for move in selected:
        board.push(move)
    return tuple(move.uci() for move in selected)


def _common_prefix(pos,neg,minimum,rng,task):
    """Bounded 4-ply knight cycles interleaved with 2-ply pawn changes.

    Every two completed knight cycles require a pawn block, preventing the
    start position of each cycle from recurring five times.  Before the
    repetition task suffix, a fresh pawn block is mandatory, so the suffix
    starts from a new position whose earlier occurrences cannot leak labels.
    """
    board=chess.Board()
    prefix=()
    cycles_since_pawn=0
    reserved=_pawn_start_squares_used_by_suffix(pos,neg)
    while len(prefix)<minimum or (task=="repetition" and cycles_since_pawn!=0):
        # For repetition, leave space for a final irreversible move pair.
        pawn_required=(cycles_since_pawn>=2 or
                       (task=="repetition" and len(prefix)>=minimum))
        if not pawn_required:
            try:
                cycle=_knight_cycle(board,rng)
                prefix+=cycle
                cycles_since_pawn+=1
                continue
            except ValueError:
                pawn_required=True
        if pawn_required:
            block=_quiet_pawn_pair(board,reserved,pos,neg,prefix,rng)
            prefix+=block
            cycles_since_pawn=0
        if len(prefix)>minimum+20:
            raise ValueError("Prefix exceeded bounded target overshoot")
    if len(prefix)%2:
        raise AssertionError("Prefix changed side to move")
    return prefix


def extend_pair(pair,target_plies=64,seed=3101,attempts=100):
    if target_plies<0 or attempts<1:
        raise ValueError("Invalid stress settings")
    task=pair["task"]
    pos=tuple(pair["positive"]["history_uci"])
    neg=tuple(pair["negative"]["history_uci"])
    if len(pos)!=len(neg):
        raise ValueError("Unequal counterfactual lengths")
    rng=random.Random(f"histchess-v4.1|{seed}|{pair['pair_id']}|{target_plies}")
    failures=[]
    for _ in range(attempts):
        try:
            if task=="castling":
                extra=max(0,target_plies-len(pos))
                add=_common_castling_suffix(pos,neg,extra,rng)
                new_pos=pos+add
                new_neg=neg+add
                protocol="synthetic_common_castling_continuation_v2"
            else:
                extra=max(0,target_plies-len(pos))
                prefix=_common_prefix(pos,neg,extra,rng,task)
                new_pos=prefix+pos
                new_neg=prefix+neg
                protocol="synthetic_knight_cycles_with_pawn_resets_v2"
            _verify_trajectory(new_pos)
            _verify_trajectory(new_neg)
            rebuilt=pair_from_moves(task,new_pos,new_neg)
            if len(new_pos)<target_plies:
                raise ValueError("Requested minimum history length not achieved")
            rebuilt["generator_family"]=pair.get("generator_family","legacy")
            rebuilt["split"]=pair.get("split","unassigned")
            rebuilt["stress_protocol"]=protocol
            rebuilt["first_divergence_distance"]=len(new_pos)-1-next(
                (i for i,(u,v) in enumerate(zip(new_pos,new_neg)) if u!=v),len(new_pos))
            return rebuilt
        except (ValueError,AssertionError) as exc:
            failures.append(str(exc))
    raise RuntimeError(f"Cannot create certified {target_plies}-ply {task} pair "
                       f"after {attempts} attempts; last failures: {failures[-3:]}")


def extend_pairs(pairs,target_plies=64,seed=3101,attempts=100):
    output=[]
    used=set()
    for i,pair in enumerate(pairs):
        for retry in range(12):
            extended=extend_pair(pair,target_plies,seed+retry*7919,attempts)
            if extended["board_key"] not in used:
                used.add(extended["board_key"])
                output.append(extended)
                break
        else:
            raise RuntimeError(f"Stress generation produced repeated observable boards: pair {i}")
    return output

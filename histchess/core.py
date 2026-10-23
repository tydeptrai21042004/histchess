"""Legal history pair construction. No pretrained engines or PGN download necessary.

All rules and move legality are checked with python-chess. Uses current-position
threefold occurrence, NOT prospective claimability (which is a different label).
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

import chess

TASKS = ("castling", "en_passant", "halfmove", "repetition")
CAST_POS = ("g1f3", "g8f6", "b1c3", "b8c6", "e2e4", "e7e5", "f3g1", "c6b8", "c3b1", "f6g8")
CAST_NEG = ("g1f3", "g8f6", "h1g1", "b8c6", "e2e4", "e7e5", "g1h1", "c6b8", "f3g1", "f6g8")
EP_POS = ("e2e4", "a7a6", "e4e5", "d7d5")
EP_NEG = ("e2e4", "d7d5", "e4e5", "a7a6")
HALF_POS = ("e2e4", "e7e5", "g1f3", "g8f6", "f3g1", "f6g8")
HALF_NEG = ("g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "e7e5")
REP_POS = ("g1f3", "g8f6", "f3g1", "f6g8") * 2
REP_NEG = ("g1f3", "g8f6", "b1c3", "b8c6", "c3b1", "c6b8", "f3g1", "f6g8")


def digest(value):
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def replay(moves):
    board = chess.Board()
    for ply, uci in enumerate(moves, 1):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {ply}: {uci}; board={board.fen()}")
        board.push(move)
    return board


def board_key(board):
    return board.board_fen() + " " + ("w" if board.turn else "b")


def position_key(board):
    """FIDE repetition equivalence, using canonical legal-en-passant FEN.

    The first four fields include piece layout, side to move, castling, and
    a legally available en-passant square. Clocks are excluded.
    """
    return " ".join(board.fen(en_passant="legal").split()[:4])


def repetitions(board):
    target = position_key(board)
    b = chess.Board()
    matches = [0] if position_key(b) == target else []
    for ply, move in enumerate(board.move_stack, 1):
        b.push(move)
        if position_key(b) == target:
            matches.append(ply)
    return matches


def observed_label(board, task):
    return bool({
        "castling": lambda: board.has_kingside_castling_rights(chess.WHITE),
        "en_passant": lambda: board.has_legal_en_passant(),
        "halfmove": lambda: board.halfmove_clock >= 2,
        "repetition": lambda: board.is_repetition(3),
    }[task]())


def annotate(board, task, history):
    rep_plies = repetitions(board)
    fen_legal = board.fen(en_passant="legal")
    return {
        "history_uci": list(history),
        "board_observation": board_key(board),
        "fen": fen_legal,
        "fen_strict": board.fen(en_passant="fen"),
        "label": int(observed_label(board, task)),
        "white_kingside_castling": bool(board.has_kingside_castling_rights(chess.WHITE)),
        "legal_en_passant": bool(board.has_legal_en_passant()),
        "halfmove_clock": int(board.halfmove_clock),
        "current_position_occurrences": len(rep_plies),
        "repetition_occurrence_plies": rep_plies,
        "legal_move_count": board.legal_moves.count(),
        "check_status": bool(board.is_check()),
        "certificate": {
            "type": "replay_from_initial_position",
            "move_count": len(history),
            "history_sha256": digest(" ".join(history)),
            "rule_observation": {"castling": "white_kingside_castling", "en_passant": "legal_en_passant", "halfmove": "halfmove_clock>=2", "repetition": "current_position_occurrences>=3"}[task],
        },
    }


def common_pawn_prefix(rng, task, rounds):
    """Generate a common, legal quiet-pawn prefix without disturbing templates."""
    b = chess.Board()
    hist = []
    for _ in range(rounds * 2):
        allowed = ("abdg h".replace(" ", "") if task == "repetition" else
                   "abc dgh".replace(" ", "") if task == "halfmove" else
                   "abcdfgh" if b.turn else "bcefgh")
        # For repetition, never advance c/f pawns into the knight landing squares.
        if task == "repetition":
            allowed = "abdegh"
        elif task == "halfmove":
            allowed = "abcdgh"
        elif b.turn:  # en passant white: retain e2 and make e4/e5 available
            allowed = "abcdfgh"
        else:  # en passant black: retain a7 and d7
            allowed = "bcefgh"
        moves = [m for m in b.legal_moves
                 if b.piece_type_at(m.from_square) == chess.PAWN
                 and chess.square_name(m.from_square)[0] in allowed
                 and not b.is_capture(m) and not m.promotion]
        if not moves:
            raise ValueError("No eligible prefix pawn moves")
        mv = rng.choice(moves)
        hist.append(mv.uci())
        b.push(mv)
    return tuple(hist)


def castling_suffix(rng, aa, bb, length):
    """Append identical moves legal in BOTH histories, preserving target label."""
    left, right = replay(aa), replay(bb)
    a, b = list(aa), list(bb)
    for _ in range(length):
        choices = []
        for move in left.legal_moves:
            if move not in right.legal_moves:
                continue
            lc = left.copy(stack=True)
            rc = right.copy(stack=True)
            lc.push(move)
            rc.push(move)
            if (lc.has_kingside_castling_rights(chess.WHITE)
                    and not rc.has_kingside_castling_rights(chess.WHITE)
                    and board_key(lc) == board_key(rc)):
                choices.append(move)
        if not choices:
            break
        m = rng.choice(choices)
        left.push(m); right.push(m)
        a.append(m.uci()); b.append(m.uci())
    return tuple(a), tuple(b)


def make_candidate(task, rng):
    if task == "castling":
        aa, bb = CAST_POS, CAST_NEG
        # A shared legal continuation diversifies observable boards.
        return castling_suffix(rng, aa, bb, rng.randint(1, 12))
    prefix = common_pawn_prefix(rng, task, rng.randint(0, 4))
    options = {
        "en_passant": (EP_POS, EP_NEG),
        "halfmove": (HALF_POS, HALF_NEG),
        "repetition": (REP_POS, REP_NEG),
    }
    pos, neg = options[task]
    return prefix + pos, prefix + neg


def pair_from_moves(task, pos, neg):
    if len(pos) != len(neg):
        raise ValueError("Counterfactual pair must have same ply count")
    a, b = replay(pos), replay(neg)
    if board_key(a) != board_key(b):
        raise ValueError(f"Observable boards differ: {board_key(a)} vs {board_key(b)}")
    if observed_label(a, task) is not True or observed_label(b, task) is not False:
        raise ValueError("Positive/negative rule labels not established")
    if task == "repetition" and a.fen(en_passant="legal") != b.fen(en_passant="legal"):
        raise ValueError("Repetition pair must have identical full FEN")
    if task in ("castling", "en_passant", "halfmove"):
        # Exactly one field of the six-field canonical FEN may change.
        field = {"castling": 2, "en_passant": 3, "halfmove": 4}[task]
        f1, f2 = a.fen(en_passant="legal").split(), b.fen(en_passant="legal").split()
        diff = [i for i, (x, y) in enumerate(zip(f1, f2)) if x != y]
        if diff != [field]:
            raise ValueError(f"Pair not single-field isolated: task={task}, differs={diff}")
    pair_id = digest(task + "|" + " ".join(pos) + "|" + " ".join(neg))[:20]
    return {
        "pair_id": pair_id,
        "task": task,
        "board_key": board_key(a),
        "positive": annotate(a, task, pos),
        "negative": annotate(b, task, neg),
    }


def generate(count_per_task=80, seed=3101, max_attempt_multiplier=100,
             generator="legacy", split_strategy="board"):
    """Construct verified history pairs.

    ``legacy`` exactly retains the original 2026 release behavior. ``diverse``
    supplies three independently defined generation families per chess rule.
    """
    if generator == "diverse":
        from .diverse import generate_diverse
        return generate_diverse(count_per_task=count_per_task, seed=seed,
                                max_attempt_multiplier=max(250, max_attempt_multiplier),
                                split_strategy=split_strategy)
    if generator != "legacy" or split_strategy != "board":
        raise ValueError("legacy supports only board splits; use generator=diverse for family splits")
    rng = random.Random(seed)
    result = []
    used = set()
    failures = collections.Counter()
    for task in TASKS:
        made = 0
        for _ in range(max_attempt_multiplier * count_per_task):
            if made >= count_per_task:
                break
            try:
                pos, neg = make_candidate(task, rng)
                pair = pair_from_moves(task, pos, neg)
                # A repeated board for a task contributes no new independent position.
                novelty_key = (task, pair["board_key"])
                if novelty_key in used:
                    failures[task+":duplicate_board"] += 1
                    continue
                used.add(novelty_key)
                result.append(pair)
                made += 1
            except (ValueError, AssertionError) as e:
                failures[task + ":" + type(e).__name__] += 1
                continue
        if made < count_per_task:
            raise RuntimeError(f"Could only make {made}/{count_per_task} unique {task} pairs; rejection counts: {dict(failures)}")
    # Deterministic group-disjoint split by observable board ACROSS tasks.
    for pair in result:
        bucket = int(digest(f"{seed}|{pair['board_key']}")[:8], 16) % 100
        pair["split"] = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
    return result, dict(failures)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(s) for s in f if s.strip()]


def rows_from_pairs(pairs):
    rows = []
    for pair in pairs:
        for variant in ("positive", "negative"):
            item = pair[variant]
            rows.append({
                "pair_id": pair["pair_id"],
                "task": pair["task"],
                "split": pair["split"],
                "variant": variant,
                "board_key": pair["board_key"],
                "generator_family": pair.get("generator_family", "legacy"),
                "history_uci": item["history_uci"],
                "fen": item["fen"],
                "label": int(item["label"]),
            })
    return rows


def validate(pairs):
    seen_ids, seen_board_tasks = set(), set()
    split_by_board = {}
    errors = []
    by_task = collections.Counter()
    by_split = collections.Counter()
    for p in pairs:
        try:
            if p["pair_id"] in seen_ids:
                raise ValueError("duplicate pair_id")
            seen_ids.add(p["pair_id"])
            if (p["task"], p["board_key"]) in seen_board_tasks:
                raise ValueError("duplicate task+board key")
            seen_board_tasks.add((p["task"], p["board_key"]))
            old = split_by_board.setdefault(p["board_key"], p["split"])
            if old != p["split"]:
                raise ValueError("board leakage across splits")
            rebuilt = pair_from_moves(p["task"], p["positive"]["history_uci"], p["negative"]["history_uci"])
            if p["pair_id"] != rebuilt["pair_id"] or p["board_key"] != rebuilt["board_key"]:
                raise ValueError("changed pairing metadata")
            for variant in ("positive", "negative"):
                if p[variant] != rebuilt[variant]:
                    raise ValueError("certificate/annotation mismatch")
            by_task[p["task"]] += 1
            by_split[p["split"]] += 1
        except Exception as exc:
            errors.append({"pair_id": p.get("pair_id"), "error": str(exc)})
    return {
        "passed": not errors and len(seen_ids) == len(pairs),
        "n_pairs": len(pairs), "n_records": len(pairs) * 2,
        "task_counts": dict(by_task), "split_counts": dict(by_split),
        "distinct_observable_boards": len(split_by_board),
        "duplicate_pair_ids": len(pairs) - len(seen_ids),
        "errors": errors[:25],
    }

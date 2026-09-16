"""Evaluation harness for the Gomoku RL challenge. DO NOT MODIFY.

Usage:
    python evaluate.py --agent student_agent
    python evaluate.py --agent student_agent --games 5   # quick pipeline check

Runs the given agent against the two fixed baselines (pure_mcts, alphazero),
splitting games evenly between them and alternating colors within each split.
Seeds are fixed, so repeated runs against a deterministic agent reproduce the
same results. Each game gives a player a 300s (5 min) cumulative wall-clock
budget for get_action() calls; exceeding it forfeits that game (sudden death).
"""
import argparse
import importlib
import random
import time

import numpy as np

import baseline_bot
from game import Board

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
N_IN_ROW = 5
TIME_LIMIT_SECONDS = 300.0
BASE_SEED = 20240101
OPPONENTS = ["pure_mcts", "alphazero"]


class TimedAgent(object):
    """Wraps an agent and tracks cumulative wall-clock spent inside get_action()."""

    def __init__(self, agent, time_limit=TIME_LIMIT_SECONDS):
        self.agent = agent
        self.time_limit = time_limit
        self.elapsed = 0.0
        self.forfeited = False

    def set_player_ind(self, p):
        self.agent.set_player_ind(p)

    def get_action(self, board):
        start = time.perf_counter()
        move = self.agent.get_action(board)
        self.elapsed += time.perf_counter() - start
        if self.elapsed > self.time_limit:
            self.forfeited = True
        return move


def play_one_game(black_agent, white_agent, seed):
    """Play one game, black moves first. Returns (winner_player_id_or_-1, reason)."""
    random.seed(seed)
    np.random.seed(seed)

    board = Board(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_in_row=N_IN_ROW)
    board.init_board(start_player=0)
    p1, p2 = board.players  # p1=1 (black, moves first), p2=2 (white)

    timed_black = TimedAgent(black_agent)
    timed_white = TimedAgent(white_agent)
    timed_black.set_player_ind(p1)
    timed_white.set_player_ind(p2)
    players = {p1: timed_black, p2: timed_white}

    while True:
        current = board.get_current_player()
        agent = players[current]
        move = agent.get_action(board)
        if agent.forfeited:
            winner = p2 if current == p1 else p1
            return winner, "forfeit(time)"
        board.do_move(move)
        end, winner = board.game_end()
        if end:
            return winner, "normal"


def run_match(student_factory, opponent_name, n_games, seed_offset, verbose=True):
    """Play n_games between the student agent and a baseline, alternating colors.

    First half: student is black (moves first). Second half: student is white.
    """
    results = []  # list of dicts: {game, student_color, outcome}
    n_black = (n_games + 1) // 2
    for i in range(n_games):
        student_is_black = i < n_black
        seed = BASE_SEED + seed_offset + i
        student_agent = student_factory()
        opponent_agent = baseline_bot.make_baseline(opponent_name)

        if student_is_black:
            winner, reason = play_one_game(student_agent, opponent_agent, seed)
            student_player_id = 1
        else:
            winner, reason = play_one_game(opponent_agent, student_agent, seed)
            student_player_id = 2

        if winner == -1:
            outcome = "draw"
        elif winner == student_player_id:
            outcome = "win"
        else:
            outcome = "loss"

        results.append(
            {
                "game": i,
                "color": "black" if student_is_black else "white",
                "outcome": outcome,
                "reason": reason,
            }
        )
        if verbose:
            print(
                "  [{}] game {:>3}/{} vs {:<10} student={:<5} -> {:<4} ({})".format(
                    opponent_name, i + 1, n_games, opponent_name,
                    "black" if student_is_black else "white", outcome, reason,
                )
            )
    return results


def summarize(results):
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    draws = sum(1 for r in results if r["outcome"] == "draw")
    total = len(results)
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    return {"wins": wins, "losses": losses, "draws": draws, "total": total, "win_rate": win_rate}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Gomoku agent against fixed baselines.")
    parser.add_argument("--agent", type=str, default="student_agent", help="module exposing make_agent()")
    parser.add_argument("--games", type=int, default=100, help="total games across both baselines")
    parser.add_argument("--quiet", action="store_true", help="suppress per-game output")
    args = parser.parse_args()

    module = importlib.import_module(args.agent)
    student_factory = module.make_agent

    n_opponents = len(OPPONENTS)
    per_opponent = [args.games // n_opponents] * n_opponents
    for i in range(args.games - sum(per_opponent)):
        per_opponent[i] += 1

    all_results = {}
    seed_offset = 0
    for opponent_name, n_games in zip(OPPONENTS, per_opponent):
        if n_games == 0:
            continue
        print("=== {} vs {} ({} games) ===".format(args.agent, opponent_name, n_games))
        results = run_match(student_factory, opponent_name, n_games, seed_offset, verbose=not args.quiet)
        seed_offset += n_games
        all_results[opponent_name] = results

    print("\n================ SUMMARY ================")
    grand_total = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
    for opponent_name, results in all_results.items():
        s = summarize(results)
        black_results = [r for r in results if r["color"] == "black"]
        white_results = [r for r in results if r["color"] == "white"]
        sb, sw = summarize(black_results), summarize(white_results)
        print(
            "{:<10} | games={:<4} W={:<3} L={:<3} D={:<3} win_rate={:.3f}"
            "  (black: {}/{} , white: {}/{})".format(
                opponent_name, s["total"], s["wins"], s["losses"], s["draws"], s["win_rate"],
                sb["wins"], sb["total"], sw["wins"], sw["total"],
            )
        )
        for k in ("wins", "losses", "draws", "total"):
            grand_total[k] += s[k]

    if grand_total["total"]:
        overall_rate = (grand_total["wins"] + 0.5 * grand_total["draws"]) / grand_total["total"]
        print(
            "\nOVERALL   | games={:<4} W={:<3} L={:<3} D={:<3} win_rate={:.3f}".format(
                grand_total["total"], grand_total["wins"], grand_total["losses"],
                grand_total["draws"], overall_rate,
            )
        )


if __name__ == "__main__":
    main()

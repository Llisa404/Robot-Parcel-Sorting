"""Round-robin league between multiple participants' agents. DO NOT MODIFY.

Usage:
    python league.py --teams team_A team_B team_C --games 20

Each pair of teams plays `--games` games, alternating colors evenly. Prints a
head-to-head win-rate matrix and a final standings table:
    score = wins + 0.5 * draws
    avg_win_rate = mean win rate across all opponents
"""
import argparse
import importlib
import itertools
import random
import time

import numpy as np

from evaluate import TimedAgent, BOARD_WIDTH, BOARD_HEIGHT, N_IN_ROW, BASE_SEED
from game import Board


def play_one_game(black_agent, white_agent, seed):
    random.seed(seed)
    np.random.seed(seed)

    board = Board(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_in_row=N_IN_ROW)
    board.init_board(start_player=0)
    p1, p2 = board.players

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


def load_factory(team_module):
    module = importlib.import_module(team_module)
    return module.make_agent


def play_match(name_a, factory_a, name_b, factory_b, n_games, seed_offset):
    """Play n_games between two teams, alternating colors. Returns per-team win/loss/draw."""
    stats = {name_a: {"wins": 0, "losses": 0, "draws": 0}, name_b: {"wins": 0, "losses": 0, "draws": 0}}
    n_a_black = (n_games + 1) // 2
    for i in range(n_games):
        a_is_black = i < n_a_black
        seed = BASE_SEED + seed_offset + i
        agent_a = factory_a()
        agent_b = factory_b()

        if a_is_black:
            winner, _ = play_one_game(agent_a, agent_b, seed)
            a_id, b_id = 1, 2
        else:
            winner, _ = play_one_game(agent_b, agent_a, seed)
            a_id, b_id = 2, 1

        if winner == -1:
            stats[name_a]["draws"] += 1
            stats[name_b]["draws"] += 1
        elif winner == a_id:
            stats[name_a]["wins"] += 1
            stats[name_b]["losses"] += 1
        else:
            stats[name_b]["wins"] += 1
            stats[name_a]["losses"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="Round-robin league between submitted agents.")
    parser.add_argument("--teams", type=str, nargs="+", required=True, help="module names, each exposing make_agent()")
    parser.add_argument("--games", type=int, default=20, help="games played per pairing")
    args = parser.parse_args()

    teams = args.teams
    factories = {t: load_factory(t) for t in teams}

    # head_to_head[a][b] = {"wins":.., "losses":.., "draws":..} from a's perspective
    head_to_head = {a: {b: None for b in teams if b != a} for a in teams}

    seed_offset = 0
    for a, b in itertools.combinations(teams, 2):
        print("=== {} vs {} ({} games) ===".format(a, b, args.games))
        stats = play_match(a, factories[a], b, factories[b], args.games, seed_offset)
        seed_offset += args.games
        head_to_head[a][b] = stats[a]
        head_to_head[b][a] = stats[b]
        print(
            "  {}: W{} L{} D{}   {}: W{} L{} D{}".format(
                a, stats[a]["wins"], stats[a]["losses"], stats[a]["draws"],
                b, stats[b]["wins"], stats[b]["losses"], stats[b]["draws"],
            )
        )

    print("\n============== HEAD-TO-HEAD WIN RATE MATRIX ==============")
    header = "{:<14}".format("") + "".join("{:<14}".format(t) for t in teams)
    print(header)
    for a in teams:
        row = "{:<14}".format(a)
        for b in teams:
            if a == b:
                row += "{:<14}".format("--")
            else:
                s = head_to_head[a][b]
                total = s["wins"] + s["losses"] + s["draws"]
                rate = (s["wins"] + 0.5 * s["draws"]) / total if total else 0.0
                row += "{:<14}".format("{:.3f}".format(rate))
        print(row)

    print("\n============== FINAL STANDINGS ==============")
    standings = []
    for t in teams:
        total_wins = total_losses = total_draws = 0
        rates = []
        for opp in teams:
            if opp == t:
                continue
            s = head_to_head[t][opp]
            total_wins += s["wins"]
            total_losses += s["losses"]
            total_draws += s["draws"]
            total_games = s["wins"] + s["losses"] + s["draws"]
            if total_games:
                rates.append((s["wins"] + 0.5 * s["draws"]) / total_games)
        score = total_wins + 0.5 * total_draws
        avg_win_rate = sum(rates) / len(rates) if rates else 0.0
        standings.append((t, score, total_wins, total_losses, total_draws, avg_win_rate))

    standings.sort(key=lambda row: (-row[1], -row[5]))
    print("{:<4}{:<14}{:<8}{:<6}{:<6}{:<6}{:<12}".format("#", "team", "score", "W", "L", "D", "avg_win_rate"))
    for rank, (t, score, w, l, d, avg_rate) in enumerate(standings, start=1):
        print("{:<4}{:<14}{:<8.1f}{:<6}{:<6}{:<6}{:<12.3f}".format(rank, t, score, w, l, d, avg_rate))


if __name__ == "__main__":
    main()

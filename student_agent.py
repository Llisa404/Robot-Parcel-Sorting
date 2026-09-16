"""My Gomoku submission.

Approach: rule-based threat detection (open three / open four / blocking) as a
positional evaluation function, combined with iterative-deepening alpha-beta
(negamax) search over a pruned candidate-move set. No neural network and no
training phase are required, which keeps the whole submission trivially
within the 4h training budget; the search itself respects the per-game
wall-clock budget via simple time management across the game's moves.

Required interface:
    make_agent() -> agent with set_player_ind(p) and get_action(board)
"""
import time

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
N_IN_ROW = 5

WIN_SCORE = 1_000_000_000

# score_for[length][open_ends]
_SCORE_TABLE = {
    1: {0: 0, 1: 5, 2: 10},
    2: {0: 0, 1: 100, 2: 500},
    3: {0: 0, 1: 2_000, 2: 50_000},
    4: {0: 0, 1: 500_000, 2: 10_000_000},
}


def _score_for(length, open_ends):
    if length >= N_IN_ROW:
        return WIN_SCORE
    return _SCORE_TABLE.get(length, {}).get(open_ends, 0)


def _other(player):
    return 1 if player == 2 else 2


def is_winning_move(states, move, player, width, height):
    """Fast check: does placing `player` at `move` complete N_IN_ROW in a row?"""
    r, c = move // width, move % width
    for dr, dc in DIRECTIONS:
        cnt = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < height and 0 <= cc < width and states.get(rr * width + cc) == player:
            cnt += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < height and 0 <= cc < width and states.get(rr * width + cc) == player:
            cnt += 1
            rr -= dr
            cc -= dc
        if cnt >= N_IN_ROW:
            return True
    return False


def _line_score_through(states, move, player, width, height):
    """Heuristic value of the run(s) that pass through `move` if `player` played there."""
    r, c = move // width, move % width
    states[move] = player
    total = 0
    for dr, dc in DIRECTIONS:
        pr, pc = r - dr, c - dc
        while 0 <= pr < height and 0 <= pc < width and states.get(pr * width + pc) == player:
            pr -= dr
            pc -= dc
        start_open = 0 <= pr < height and 0 <= pc < width and states.get(pr * width + pc) is None
        rr, cc = pr + dr, pc + dc
        length = 0
        while 0 <= rr < height and 0 <= cc < width and states.get(rr * width + cc) == player:
            length += 1
            rr += dr
            cc += dc
        end_open = 0 <= rr < height and 0 <= cc < width and states.get(rr * width + cc) is None
        total += _score_for(length, int(start_open) + int(end_open))
    del states[move]
    return total


def _evaluate_player(states, player, width, height):
    """Full board static evaluation for one player (sum over all their runs)."""
    total = 0
    for pos, pl in states.items():
        if pl != player:
            continue
        r, c = pos // width, pos % width
        for dr, dc in DIRECTIONS:
            pr, pc = r - dr, c - dc
            if states.get(pr * width + pc) == player and 0 <= pr < height and 0 <= pc < width:
                continue  # not a run head
            length = 0
            rr, cc = r, c
            while 0 <= rr < height and 0 <= cc < width and states.get(rr * width + cc) == player:
                length += 1
                rr += dr
                cc += dc
            end_open = 0 <= rr < height and 0 <= cc < width and states.get(rr * width + cc) is None
            start_open = 0 <= pr < height and 0 <= pc < width and states.get(pr * width + pc) is None
            total += _score_for(length, int(start_open) + int(end_open))
    return total


def _candidate_moves(states, availables, width, height, radius=2):
    if not states:
        center = (height // 2 - 1) * width + (width // 2 - 1)
        return [center] if center in availables else list(availables[:1])
    avail_set = set(availables)
    candidates = set()
    for pos in states:
        r, c = pos // width, pos % width
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    m = rr * width + cc
                    if m in avail_set:
                        candidates.add(m)
    if not candidates:
        return list(availables)
    return list(candidates)


class _Search(object):
    def __init__(self, width, height, deadline):
        self.width = width
        self.height = height
        self.deadline = deadline
        self.nodes = 0

    def order_moves(self, states, moves, player, opponent):
        scored = []
        for m in moves:
            my_gain = _line_score_through(states, m, player, self.width, self.height)
            opp_gain = _line_score_through(states, m, opponent, self.width, self.height)
            scored.append((max(my_gain, opp_gain), m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def negamax(self, states, avail, to_move, depth, alpha, beta, last_move, last_mover, top_k):
        self.nodes += 1
        if last_move is not None and is_winning_move(states, last_move, last_mover, self.width, self.height):
            return -(WIN_SCORE - (10 - depth))
        if not avail:
            return 0
        if depth <= 0 or time.perf_counter() > self.deadline:
            opp = _other(to_move)
            return _evaluate_player(states, to_move, self.width, self.height) - _evaluate_player(
                states, opp, self.width, self.height
            )

        moves = _candidate_moves(states, avail, self.width, self.height)
        opponent = _other(to_move)
        moves = self.order_moves(states, moves, to_move, opponent)[:top_k]

        best = -WIN_SCORE * 2
        avail_set = avail
        for m in moves:
            states[m] = to_move
            avail_set.discard(m)
            val = -self.negamax(
                states, avail_set, opponent, depth - 1, -beta, -alpha, m, to_move, max(4, top_k - 2)
            )
            del states[m]
            avail_set.add(m)

            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
            if time.perf_counter() > self.deadline:
                break
        return best


class ThreatSearchAgent(object):
    """Heuristic evaluation + iterative-deepening alpha-beta search."""

    def __init__(self, total_time_budget=280.0, max_depth=5, root_top_k=14):
        self.player = None
        self.opponent = None
        self.total_time_budget = total_time_budget
        self.time_used = 0.0
        self.max_depth = max_depth
        self.root_top_k = root_top_k

    def set_player_ind(self, p):
        self.player = p
        self.opponent = _other(p)

    def _time_for_this_move(self, board):
        moves_left = max(1, (len(board.availables) + 1) // 2)
        remaining = max(0.5, self.total_time_budget - self.time_used)
        budget = remaining / moves_left
        return max(0.2, min(10.0, budget * 1.4))

    def get_action(self, board):
        start = time.perf_counter()
        width, height = board.width, board.height
        states = dict(board.states)
        availables = board.availables

        move = self._pick_move(states, availables, width, height, self._time_for_this_move(board))

        self.time_used += time.perf_counter() - start
        return move

    def _pick_move(self, states, availables, width, height, time_budget):
        if len(availables) == width * height:
            center = (height // 2 - 1) * width + (width // 2 - 1)
            return center if center in availables else availables[0]

        for m in availables:
            if is_winning_move(states, m, self.player, width, height):
                return m
        for m in availables:
            if is_winning_move(states, m, self.opponent, width, height):
                return m

        candidates = _candidate_moves(states, availables, width, height)
        if len(candidates) == 1:
            return candidates[0]

        deadline = time.perf_counter() + time_budget
        best_move = candidates[0]
        avail_set = set(availables)

        depth = 2
        while depth <= self.max_depth and time.perf_counter() < deadline:
            search = _Search(width, height, deadline)
            ordered = search.order_moves(states, candidates, self.player, self.opponent)[: self.root_top_k]

            local_best_move = None
            local_best_val = -WIN_SCORE * 2
            alpha, beta = -WIN_SCORE * 2, WIN_SCORE * 2
            for m in ordered:
                states[m] = self.player
                avail_set.discard(m)
                val = -search.negamax(
                    states, avail_set, self.opponent, depth - 1, -beta, -alpha, m, self.player,
                    max(4, self.root_top_k - 2),
                )
                del states[m]
                avail_set.add(m)

                if val > local_best_val:
                    local_best_val = val
                    local_best_move = m
                if local_best_val > alpha:
                    alpha = local_best_val
                if time.perf_counter() > deadline:
                    break

            if local_best_move is not None:
                best_move = local_best_move
                if local_best_val >= WIN_SCORE - 100:
                    break
            depth += 1

        return best_move

    def __str__(self):
        return "ThreatSearchAgent {}".format(self.player)


def make_agent():
    return ThreatSearchAgent()

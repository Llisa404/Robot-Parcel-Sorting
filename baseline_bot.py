"""Fixed baseline opponents for the challenge. DO NOT MODIFY.

Two deterministic baselines:
  - "pure_mcts": classic MCTS with no neural network, n_playout=1000.
  - "alphazero": pretrained policy/value net + MCTS, n_playout=400.

Both baselines play deterministically, so results are reproducible.
"""
import os

from mcts_pure import MCTSPlayer as PureMCTSPlayer

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_policy_8_8_5.model")

_alphazero_cache = None


def _build_pure_mcts():
    return PureMCTSPlayer(c_puct=5, n_playout=1000)


def _build_alphazero():
    global _alphazero_cache
    if _alphazero_cache is not None:
        return _alphazero_cache
    from policy_value_net_pytorch import PolicyValueNet
    from mcts_alphaZero import MCTSPlayer as AlphaZeroMCTSPlayer

    policy_value_net = PolicyValueNet(BOARD_WIDTH, BOARD_HEIGHT, model_file=MODEL_FILE)
    player = AlphaZeroMCTSPlayer(
        policy_value_net.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0
    )
    _alphazero_cache = player
    return player


_BUILDERS = {
    "pure_mcts": _build_pure_mcts,
    "alphazero": _build_alphazero,
}


def make_baseline(name):
    """Return a fresh baseline agent instance by name ('pure_mcts' or 'alphazero')."""
    if name not in _BUILDERS:
        raise ValueError("Unknown baseline '{}'. Choices: {}".format(name, list(_BUILDERS)))
    return _BUILDERS[name]()


def available_baselines():
    return list(_BUILDERS.keys())

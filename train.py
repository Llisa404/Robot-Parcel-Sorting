"""Self-play training pipeline for an AlphaZero-style policy-value net.

Used to (a) produce the pretrained weights behind the "alphazero" baseline
(best_policy_8_8_5.model) and (b) serve as a starting point for participants
who want to train their own self-play agent for student_agent.py.

Usage:
    python train.py --game-batch-num 1000 --time-budget-min 90 \
        --output my_policy_8_8_5.model
"""
import argparse
import random
import time
from collections import defaultdict, deque

import numpy as np

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer as MCTS_AlphaZero
from policy_value_net_pytorch import PolicyValueNet

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
N_IN_ROW = 5


class TrainPipeline(object):
    def __init__(self, init_model=None, output_path="best_policy_8_8_5.model"):
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.n_in_row = N_IN_ROW
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 256
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.output_path = output_path

        self.pure_mcts_playout_num = 200
        self.best_win_ratio = 0.0

        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        self.mcts_player = MCTS_AlphaZero(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

    def get_equi_data(self, play_data):
        """Augment the data set with board rotations/flips (8x symmetry of the board)."""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in (1, 2, 3, 4):
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i
                )
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [d[0] for d in mini_batch]
        mcts_probs_batch = [d[1] for d in mini_batch]
        winner_batch = [d[2] for d in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        loss = entropy = None
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(
            "kl:{:.5f}, lr_multiplier:{:.3f}, loss:{}, entropy:{}".format(
                kl, self.lr_multiplier, loss, entropy
            )
        )
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTS_AlphaZero(
            self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout
        )
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0
            )
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(
            "num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]
            )
        )
        return win_ratio

    def run(self, game_batch_num, time_budget_min=None):
        deadline = time.time() + time_budget_min * 60 if time_budget_min else None
        try:
            for i in range(game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()

                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate(n_games=8)
                    self.policy_value_net.save_model("current_policy_8_8_5.model")
                    if win_ratio > self.best_win_ratio:
                        print("New best policy, win_ratio={:.3f}".format(win_ratio))
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(self.output_path)
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                if deadline and time.time() > deadline:
                    print("Time budget reached, stopping at batch {}".format(i + 1))
                    break
        except KeyboardInterrupt:
            print("\n\rquit")

        # Make sure a model file always exists at the output path.
        if self.best_win_ratio == 0.0:
            self.policy_value_net.save_model(self.output_path)


def main():
    parser = argparse.ArgumentParser(description="Train an AlphaZero-style Gomoku agent via self-play.")
    parser.add_argument("--game-batch-num", type=int, default=1500)
    parser.add_argument("--time-budget-min", type=float, default=None, help="stop after this many minutes")
    parser.add_argument("--init-model", type=str, default=None)
    parser.add_argument("--output", type=str, default="best_policy_8_8_5.model")
    parser.add_argument("--n-playout", type=int, default=400)
    parser.add_argument("--check-freq", type=int, default=50)
    args = parser.parse_args()

    pipeline = TrainPipeline(init_model=args.init_model, output_path=args.output)
    pipeline.n_playout = args.n_playout
    pipeline.mcts_player = MCTS_AlphaZero(
        pipeline.policy_value_net.policy_value_fn,
        c_puct=pipeline.c_puct,
        n_playout=pipeline.n_playout,
        is_selfplay=1,
    )
    pipeline.check_freq = args.check_freq
    pipeline.run(args.game_batch_num, time_budget_min=args.time_budget_min)


if __name__ == "__main__":
    main()

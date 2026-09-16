"""PyTorch policy-value network: conv trunk with a policy head and a value head."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common conv trunk
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # policy head
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # value head
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet(object):
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4

        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(), weight_decay=self.l2_const
        )

        if model_file:
            state_dict = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(state_dict)

    def policy_value(self, state_batch):
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """Given a game.Board, return ((action, prob) pairs, value) for available moves."""
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, 4, self.board_width, self.board_height)
        )
        state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = float(value.item())
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        mcts_probs = torch.tensor(np.array(mcts_probs), dtype=torch.float32, device=self.device)
        winner_batch = torch.tensor(np.array(winner_batch), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        torch.save(self.get_policy_param(), model_file)

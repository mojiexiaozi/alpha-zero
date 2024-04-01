import torch
from torch import nn
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from typing import Tuple

from alpha_zero.neural_net import NeuralNet
from alpha_zero.gobang.game import Game
from alpha_zero.utils import AverageMeter


class GobangNet(nn.Module):
    def __init__(
        self,
        game: Game,
        num_channels: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        self.block = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_channels * (self.board_x - 4) * (self.board_y - 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pi = nn.Sequential(nn.Linear(512, self.action_size), nn.Softmax(dim=1))
        self.v = nn.Sequential(nn.Linear(512, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 1, self.board_x, self.board_y)
        x = self.block(x)
        pi = self.pi(x)
        return pi, self.v(x)


class GobangNetWrapper(NeuralNet):
    def __init__(self, game: Game) -> None:
        self.net = GobangNet(game)
        self.board_x, self.board_y = game.get_board_size()
        self.backup_state = None
        if torch.cuda.is_available():
            self.net.cuda()

    def train(self, examples, epochs=100, batch_size=64) -> None:
        optimizer = torch.optim.Adam(self.net.parameters())

        batch_size = min(batch_size, len(examples))
        batch_count = int(len(examples) / batch_size)
        for epoch in range(epochs):
            pabr = tqdm(range(batch_count), desc=f"Epoch {epoch + 1}/{epochs}")
            self.net.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            for _ in pabr:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                pieces, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                pieces_tensor = torch.FloatTensor(np.array(pieces).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                if torch.cuda.is_available():
                    pieces_tensor = pieces_tensor.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # predict
                pred_pis, pred_vs = self.net(pieces_tensor)

                # calculate loss
                loss_pi = -torch.sum(target_pis * pred_pis) / target_pis.size()[0]
                loss_v = (
                    torch.sum((target_vs - pred_vs.view(-1)) ** 2) / target_vs.size()[0]
                )
                total_loss = loss_pi + loss_v
                pi_losses.update(loss_pi.item(), target_pis.size(0))
                v_losses.update(loss_v.item(), target_vs.size(0))

                # update progress bar
                pabr.set_postfix(pi_loss=pi_losses, v_loss=v_losses)

                # backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, pieces: np.ndarray):
        pieces_tensor = torch.FloatTensor(pieces.astype(np.float64))
        if torch.cuda.is_available():
            pieces_tensor = pieces_tensor.contiguous().cuda()

        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(pieces_tensor)
        return pi.cpu().numpy()[0], v.cpu().numpy()[0]

    def backup(self) -> None:
        self.backup_state = deepcopy(self.net.state_dict())

    def restore(self) -> None:
        self.net.load_state_dict(self.backup_state)

    def load_from_net(self, net) -> None:
        self.net.load_state_dict(net.net.state_dict())


if __name__ == "__main__":
    from alpha_zero.gobang.game import GobangGame

    game = GobangGame(10, 5)
    net = GobangNetWrapper(game)
    board = game.get_init_board()
    pieces = board.get_pieces()
    pi, v = net.predict(pieces)

    net.train([(pieces, pi, v)], epochs=10)

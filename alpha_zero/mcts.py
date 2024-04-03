import logging
import math
import numpy as np

from alpha_zero.env.game import Game
from alpha_zero.neural_net import NeuralNet
from alpha_zero.utils import Config

EPS = 1e-8

LOGGER = logging.getLogger(__name__)


class MCTS:
    def __init__(self, game: Game, net: NeuralNet, args=Config()):
        self.game = game
        self.net = net
        self.args = args
        self.Qsa = {}  # 存储动作价值
        self.Nsa = {}  # 存储动作访问次数
        self.Ns = {}  # 存储状态访问次数
        self.Ps = {}  # 存储动作概率

        self.Es = {}  # 存储游戏结束状态
        self.Vs = {}  # 存储游戏有效状态

    def search(self, canonical_board):
        """执行一次蒙卡洛树搜索, 返回最终叶子节点的价值"""
        s = self.game.string_representation(canonical_board)

        # 存储游戏状态
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)

        # 游戏结束
        if self.Es[s] != 0:
            return -self.Es[s]

        # 未探索过的状态
        if s not in self.Ps:
            # 叶节点，使用神经网络预测动作概率和状态价值
            self.Ps[s], v = self.net.predict(canonical_board)
            valid_actions = self.game.get_valid_actions(canonical_board, 1)
            self.Ps[s] = self.Ps[s] * valid_actions
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                LOGGER.error("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valid_actions
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valid_actions
            self.Ns[s] = 0
            return -v

        valid_actions = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # UCB算法选择动作
        for a in range(self.game.get_action_size()):
            if not valid_actions[a]:
                continue
            # 有效步
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, a)]
                )
            else:
                u = self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_board(next_s, next_player)

        # 递归搜索
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v

    def get_action_prob(self, canonical_board, temp=1):
        """执行num_mcts_sims次蒙卡洛树搜索, 返回动作概率"""
        for i in range(self.args.num_mcts_sims):
            self.search(canonical_board)

        s = self.game.string_representation(canonical_board)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.get_action_size())
        ]
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            prob_list = [0] * len(counts)
            prob_list[best_a] = 1
            return prob_list
        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        prob_list = [x / counts_sum for x in counts]
        return prob_list

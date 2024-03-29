from tqdm import tqdm
import numpy as np
from collections import deque

from alpha_zero.game import Game
from alpha_zero.neural_net import NeuralNet
from alpha_zero.mcts import MCTS


class Coach:
    def __init__(self, game: Game, net: NeuralNet, args) -> None:
        self.game = game
        self.net = net
        self.competitor_net = self.net.__class__(self.game)  # 竞争对手net
        self.args = args
        self.mcts = MCTS(game, net, args)
        self.train_examples_history = []
        self.skip_first_self_play = False

    def execute(self):
        train_examples = []
        board = self.game.get_init_board()
        cur_player = 1
        step = 0

        while True:
            step += 1
            canonical_board = self.game.get_canonical_board(board, cur_player)
            temp = int(step < self.args.temp_threshold)

            pi = self.mcts.get_action_prob(canonical_board, temp=temp)
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.get_next_state(board, cur_player, action)

            r = self.game.get_game_ended(board, cur_player)
            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != cur_player)))
                    for x in train_examples
                ]

    def learn(self, num_iters=100):
        for i in range(1, num_iters + 1):
            print(f"Iter {i}/{num_iters}")
            if not self.skip_first_self_play or i > 1:
                iter_examples = deque([], maxlen=self.args.maxlen_of_queue)
                for _ in tqdm(range(self.args.num_episodes), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.net, self.args)
                    iter_examples += self.execute()
                self.train_examples_history.append(iter_examples)

            if len(self.train_examples_history) > self.args.num_iters_for_training:
                print("Removing the oldest train examples...")
                self.train_examples_history.pop(0)

            self.save_train_examples(i - 1)

            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            np.random.shuffle(train_examples)

            self.net.backup()
            self.competitor_net.load_from_net(self.net)
            competitor_mcts = MCTS(self.game, self.competitor_net, self.args)

            self.net.train(train_examples)
            mcts = MCTS(self.game, self.net, self.args)

            print("PITTING AGAINST PREVIOUS VERSION")
            

    def save_train_examples(self, iter):
        examples = []
        for e in self.train_examples_history:
            examples.extend(e)
        np.save(f"train_examples_{iter}.npy", examples)

    def load_train_examples(self, iter):
        examples = np.load(f"train_examples_{iter}.npy")
        self.train_examples_history.append(examples)
        self.skip_first_self_play = True

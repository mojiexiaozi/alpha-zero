import numpy as np

from alpha_zero.player import Player
from alpha_zero.gobang.game import Game, GobangBoard


class RandomPlayer(Player):
    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def play(self, board: GobangBoard) -> int:
        a = np.random.randint(self.game.get_action_size())
        valid_actions = self.game.get_valid_actions(board, 1)
        while valid_actions[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a

    def __call__(self, board: GobangBoard) -> int:
        return self.play(board)


if __name__ == "__main__":
    from alpha_zero.gobang.game import GobangGame
    from alpha_zero.arena import Arena
    from alpha_zero.mcts import MCTS

    game = GobangGame(6, 3)
    player1 = RandomPlayer(game)
    player2 = RandomPlayer(game)

    arena = Arena(player1, player2, game, display=GobangGame.display)

    print(arena.play_games(200, verbose=False))

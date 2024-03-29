from alpha_zero.game import Game
from alpha_zero.board import Board


class Player:
    def __init__(self, game: Game) -> None:
        self.game = game

    def play(self, board: Board) -> int:
        raise NotImplementedError

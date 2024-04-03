import numpy as np

from alpha_zero.env.board import Board


class GobangBoard(Board):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.pieces = np.zeros((n, n), dtype=int)

    def __getitem__(self, index) -> int:
        return self.pieces[index]

    def get_legal_moves(self):
        """所有合法的走法"""
        zero_indices = np.where(self.pieces == 0)
        return list(zip(zero_indices[0], zero_indices[1]))

    def has_legal_moves(self):
        """是否还有合法的走法"""
        return (self.pieces == 0).any()

    def execute_move(self, move, color: int):
        """执行走法 0: 空, -1: 黑, 1: 白"""
        x, y = move
        assert self.pieces[x, y] == 0
        self.pieces[x, y] = color

    def get_pieces(self) -> np.ndarray:
        """返回棋盘"""
        return self.pieces


if __name__ == "__main__":
    board = GobangBoard(3)
    print(board.has_legal_moves())
    board.execute_move((0, 0), -1)
    board.execute_move((0, 1), 1)
    print(board.get_legal_moves())

import numpy as np


class Board:
    def __init__(self) -> None:
        pass

    def get_legal_moves(self):
        """所有合法的走法"""
        pass

    def has_legal_moves(self):
        """是否还有合法的走法"""
        pass

    def execute_move(self, move, color: int):
        """执行走法"""
        pass

    def get_pieces(self) -> np.ndarray:
        """返回棋盘"""
        pass

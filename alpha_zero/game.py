from typing import Tuple

from alpha_zero.board import Board


class Game:
    def __init__(self):
        """初始化游戏"""
        pass

    def get_init_board(self) -> Board:
        """返回初始棋盘"""
        pass

    def get_board_size(self) -> Tuple[int, int]:
        """返回棋盘大小"""
        pass

    def get_action_size(self) -> int:
        """返回动作空间大小"""
        pass

    def get_next_state(self, board: Board, player: int, action) -> Tuple[Board, int]:
        """返回下一个状态和棋手"""
        pass

    def get_valid_moves(self, board: Board, player: int):
        """返回合法走法"""
        pass

    def get_game_ended(self, board: Board, player: int) -> int:
        """返回游戏是否结束, 0: 游戏未结束, 1: player, -1: player失败"""
        pass

    def get_canonical_form(self, board: Board, player: int) -> Board:
        """返回规范形式"""
        pass

    def get_symmetries(self, board: Board, pi) -> Tuple[Board, list]:
        """返回对称形式"""
        pass

    def string_representation(self, board: Board) -> str:
        """返回棋盘的字符串表示"""
        pass

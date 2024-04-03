from typing import Tuple
from copy import deepcopy
import numpy as np

from alpha_zero.env.game import Game
from alpha_zero.gobang.board import GobangBoard


class GobangGame(Game):
    def __init__(self, n=15, nir=5):
        """初始化游戏"""
        super().__init__()
        self.n = n
        self.n_in_row = nir

    def get_init_board(self) -> GobangBoard:
        """返回初始棋盘"""
        return GobangBoard(self.n)

    def get_board_size(self) -> Tuple[int, int]:
        """返回棋盘大小"""
        return self.n, self.n

    def get_action_size(self) -> int:
        """返回动作空间大小"""
        return self.n * self.n + 1

    def get_next_state(
        self, board: GobangBoard, player: int, action
    ) -> Tuple[GobangBoard, int]:
        """返回下一个状态和棋手"""
        if action == self.n * self.n:
            return board, -player

        b = deepcopy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return b, -player

    def get_valid_actions(self, board: GobangBoard, player: int):
        """返回合法走法"""
        vaild_actions = [0] * self.get_action_size()
        leagal_moves = board.get_legal_moves()
        if len(leagal_moves) == 0:
            vaild_actions[-1] = 1
            return np.array(vaild_actions)

        for x, y in leagal_moves:
            vaild_actions[self.n * x + y] = 1
        return np.array(vaild_actions)

    def get_game_ended(self, board: GobangBoard, player: int) -> int:
        """返回游戏是否结束, 0: 游戏未结束, 1: player, -1: player失败"""
        nir = self.n_in_row
        for w in range(self.n):
            for h in range(self.n):
                # 横向
                if (
                    w in range(self.n - nir + 1)
                    and board[w, h] != 0
                    and len(set(board[i, h] for i in range(w, w + nir))) == 1
                ):
                    return board[w, h]

                # 纵向
                if (
                    h in range(self.n - nir + 1)
                    and board[w, h] != 0
                    and len(set(board[w, j] for j in range(h, h + nir))) == 1
                ):
                    return board[w, h]

                # 斜向
                if (
                    w in range(self.n - nir + 1)
                    and h in range(self.n - nir + 1)
                    and board[w, h] != 0
                    and len(set(board[w + k, h + k] for k in range(nir))) == 1
                ):
                    return board[w, h]

                # 反斜向
                if (
                    w in range(self.n - nir + 1)
                    and h in range(nir - 1, self.n)
                    and board[w, h] != 0
                    and len(set(board[w + l, h - l] for l in range(nir))) == 1
                ):
                    return board[w, h]
        if board.has_legal_moves():
            return 0
        return 1e-4

    def get_canonical_board(self, board: GobangBoard, player: int) -> GobangBoard:
        """返回规范形式"""
        board.pieces *= player
        return board

    def get_symmetries(
        self, board: GobangBoard, pi: np.ndarray
    ) -> Tuple[GobangBoard, list]:
        """返回对称形式"""
        assert len(pi) == self.n**2 + 1
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                new_b = board.pieces.rotate(i, j)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                l += [(new_b, list(new_pi.ravel()) + [pi[-1]])]
        return l

    def string_representation(self, board: GobangBoard) -> str:
        """返回棋盘的字符串表示"""
        return board.pieces.tobytes()

    @staticmethod
    def display(board: GobangBoard):
        """显示棋盘"""
        n = board.pieces.shape[0]
        for y in range(n):
            print(y, "|", end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board.pieces[y, x]  # get the piece to print
                if piece == -1:
                    print("o ", end="")
                elif piece == 1:
                    print("x ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("   -----------------------")


if __name__ == "__main__":
    game = GobangGame(n=5)
    board = game.get_init_board()
    board.execute_move((0, 0), 1)
    board.execute_move((1, 1), -1)
    game.display(board)
    print(game.string_representation(board))
    print(game.get_game_ended(board, 1))
    print(board.get_legal_moves())
    print(game.get_valid_actions(board, 1))

    # print(game.get_valid_actions(board, 1))
    # print(game.get_action_size())
    # print(game.get_board_size())
    # print(game.get_next_state(board, 1, 0))
    # print(game.get_canonical_board(board, 1))
    # print(game.get_symmetries(board, 1))

from tqdm import tqdm

from alpha_zero.game import Game


class Arena:
    def __init__(self, player1: int, player2: int, game: Game, display=None) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.get_init_board()
        iter = 0

        while self.game.get_game_ended(board, cur_player) == 0:
            iter += 1
            # if verbose and self.display:
            #     print(f"Turn {iter} player: {cur_player}")
            #     self.display(board)
            action = players[cur_player + 1](
                self.game.get_canonical_board(board, cur_player)
            )
            valid_actions = self.game.get_valid_actions(
                self.game.get_canonical_board(board, cur_player), 1
            )
            assert valid_actions[action] > 0, f"action: {action} is not valid."

            board, cur_player = self.game.get_next_state(board, cur_player, action)
        if verbose and self.display:
            print(
                "Game over: Turn ",
                iter,
                "Cur Player ",
                cur_player,
                "Result ",
                self.game.get_game_ended(board, cur_player),
            )
            self.display(board)
        return self.game.get_game_ended(board, cur_player)

    def play_games(self, num: int, verbose=False):
        num = int(num / 2)
        one_wins = 0  # player1 wins
        two_wins = 0  # player2 wins
        draws = 0  # games ending in a draw
        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            res = self.play_game(verbose)
            if res == 1:
                one_wins += 1
            elif res == -1:
                two_wins += 1
            else:
                draws += 1
        # 对调玩家
        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            res = self.play_game(verbose)
            if res == -1:
                one_wins += 1
            elif res == 1:
                two_wins += 1
            else:
                draws += 1
        return one_wins, two_wins, draws

"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = sum(row.count(X) for row in board)
    o = sum(row.count(O) for row in board)
    return X if x == o else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    remaining_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                remaining_actions.add((i, j))
    return remaining_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] != EMPTY:
        raise Exception("Invalid action/task")
    new_board = [row[:] for row in board]
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    rows = board + [[board[i][j] for i in range(3)] for j in range(3)] + [[board[0][0], board[1][1], board[2][2]], [board[0][2], board[1][1], board[2][0]]]
    for row in rows:
        if row.count(X) == 3:
            return X
        elif row.count(O) == 3:
            return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    return all(cell is not EMPTY for row in board for cell in row)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        value = -math.inf
        upcoming_action = None
        for action in actions(board):
            min_value1 = min_value(result(board, action))
            if min_value1 > value:
                value = min_value1
                upcoming_action = action
    else:
        value = math.inf
        upcoming_action = None
        for action in actions(board):
            max_value1 = max_value(result(board, action))
            if max_value1 < value:
                value = max_value1
                upcoming_action = action

    return upcoming_action


def max_value(board):
    if terminal(board):
        return utility(board)
    minimax = -math.inf
    for action in actions(board):
        minimax = max(minimax, min_value(result(board, action)))
    return minimax


def min_value(board):
    if terminal(board):
        return utility(board)
    minimax = math.inf
    for action in actions(board):
        minimax = min(minimax, max_value(result(board, action)))
    return minimax

import random


def legal_state(board):
    # Check if the board state is valid for each row, column, and 3x3 box
    for row in board:
        seen_row = []
        for element in row:
            if element in seen_row:
                return False
            elif element != 0:
                seen_row.append(element)

    for i in range(9):
        seen_column = []
        for k in range(9):
            if board[k][i] in seen_column:
                return False
            elif board[k][i] != 0:
                seen_column.append(board[k][i])

    for box_row_start in range(0, 9, 3):
        for box_column_start in range(0, 9, 3):
            seen_box = []
            for i in range(box_row_start, box_row_start + 3):
                for j in range(box_column_start, box_column_start + 3):
                    if board[i][j] != 0:
                        if board[i][j] in seen_box:
                            return False
                        seen_box.append(board[i][j])
    return True


def generate_sudoku():
    # Generates a basic random Sudoku grid
    board = [[0] * 9 for _ in range(9)]
    # [Include your board generation logic here]
    return board


def solve_sudoku(board):
    # Fills in the Sudoku board with a solution
    # [Add solving logic if available in your original code]
    return board


def check_solution(user_board):
    # Validates the user's solution against the generated Sudoku puzzle
    return legal_state(user_board)

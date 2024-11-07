

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def is_safe(board, row, col, N):
    for i in range(row):
        if board[i][col] == 'Q':
            return False

    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j -= 1

    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j += 1

    return True

def solve_n_queens(board, row, N):
    if row == N:
        print_board(board)
        return True  # To find one solution, you can return here

    for col in range(N):
        if is_safe(board, row, col, N):
            board[row][col] = 'Q'  # Place Queen
            if solve_n_queens(board, row + 1, N):
                return True  # If solution is found, return
            board[row][col] = '.'  # Backtrack

    return False  # Trigger backtracking

def n_queens(N, first_row=0, first_col=0):
    board = [['.' for _ in range(N)] for _ in range(N)]

    board[first_row][first_col] = 'Q'
    print(f"Placing first Queen at row {first_row + 1}, column {first_col + 1}:")
    print_board(board)

    if not solve_n_queens(board, first_row + 1, N):
        print("No solution exists.")
    else:
        print("One of the possible solutions:")

if __name__== "__main__":
    N = 8  # Size of the board (8x8 for the classic problem)
    first_row = 0  # First Queen's row (0-indexed)
    first_col = 0  # First Queen's column (0-indexed)
    n_queens(N, first_row, first_col)
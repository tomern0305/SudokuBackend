# solver.py
from typing import List
import copy

def isSafe(mat: List[List[int]], row: int, col: int, num: int) -> bool:
    for x in range(9):
        if mat[row][x] == num:
            return False
    for x in range(9):
        if mat[x][col] == num:
            return False
    startRow = row - (row % 3)
    startCol = col - (col % 3)
    for i in range(3):
        for j in range(3):
            if mat[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudokuRec(mat: List[List[int]], row: int = 0, col: int = 0) -> bool:
    if row == 8 and col == 9:
        return True
    if col == 9:
        row += 1
        col = 0
    if mat[row][col] != 0:
        return solveSudokuRec(mat, row, col + 1)
    for num in range(1, 10):
        if isSafe(mat, row, col, num):
            mat[row][col] = num
            if solveSudokuRec(mat, row, col + 1):
                return True
            mat[row][col] = 0
    return False

def vector_to_matrix(vec: List[int]) -> List[List[int]]:
    return [vec[i*9:(i+1)*9] for i in range(9)]

def matrix_to_vector(mat: List[List[int]]) -> List[int]:
    return [x for row in mat for x in row]

def no_conflicts_in_given_clues(mat: List[List[int]]) -> bool:
    for i in range(9):
        row_vals = [v for v in mat[i] if v != 0]
        col_vals = [mat[r][i] for r in range(9) if mat[r][i] != 0]
        if len(row_vals) != len(set(row_vals)):
            return False
        if len(col_vals) != len(set(col_vals)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = []
            for r in range(br, br+3):
                for c in range(bc, bc+3):
                    v = mat[r][c]
                    if v != 0:
                        box.append(v)
            if len(box) != len(set(box)):
                return False
    return True

def solve_vector(vec: List[int]) -> List[int]:
    mat = vector_to_matrix(vec)
    if not no_conflicts_in_given_clues(mat):
        raise ValueError("Invalid puzzle: conflicts in given clues.")
    work = copy.deepcopy(mat)
    if not solveSudokuRec(work):
        raise ValueError("Unsolvable puzzle.")
    return matrix_to_vector(work)

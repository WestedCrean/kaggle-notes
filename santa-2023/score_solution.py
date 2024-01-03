"""Evaluation metric for Santa 2023."""
import os
import sys
import time
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict, List

import numba
import numpy as np
import polars as pl
import tqdm
from sympy.combinatorics import Permutation


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pl.DataFrame,  # a dataframe with columns: ['id', 'puzzle_type', 'solution_state', 'initial_state', 'num_wildcards']
    submission: pl.DataFrame,  # a dataframe with columns: ['id', 'moves']
    series_id_column_name: str,
    moves_column_name: str,
    puzzle_info_path: str,
) -> float:
    """Santa 2023 evaluation metric.

    Parameters
    ----------
    solution : pl.DataFrame

    submission : pl.DataFrame

    series_id_column_name : str

    moves_column_name : str

    Returns
    -------
    total_num_moves : int
    """

    puzzle_info = pl.read_csv(puzzle_info_path)
    solution_ids = solution[series_id_column_name].to_numpy()
    solution_states = solution["solution_state"].to_numpy()
    initial_states = solution["initial_state"].to_numpy()
    num_wildcards = solution["num_wildcards"].to_numpy()
    submission_ids = submission[series_id_column_name].to_numpy()
    submission_moves = submission[moves_column_name].to_numpy()

    total_num_moves = 0

    times = {}

    # check which solution_ids are not in puzzle_info
    for sol_id, sub_id, sol_state, init_state, num_wild, sub_moves in tqdm.tqdm(
        zip(
            solution_ids,
            submission_ids,
            solution_states,
            initial_states,
            num_wildcards,
            submission_moves,
        ),
        total=len(submission),
    ):
        puzzle_type = (
            solution.filter(pl.col("id") == sol_id).select("puzzle_type").to_series()[0]
        )
        allowed_moves = {}

        try:
            allowed_moves_row = (
                puzzle_info.filter(pl.col("puzzle_type") == puzzle_type)
                .select("allowed_moves")
                .to_series()[0]
            )
            allowed_moves = literal_eval(allowed_moves_row)
        except KeyError:
            print("looking up puzzle info for", sol_id, "failed")

        allowed_moves_nb = numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.ListType(numba.types.int64),
        )

        for k, v in allowed_moves.items():
            allowed_moves_nb[k] = numba.typed.List(v)

        try:
            puzzle = Puzzle(
                puzzle_id=str(sol_id),
                allowed_moves=allowed_moves_nb,
                solution_state=numba.typed.List(sol_state.split(";")),
                initial_state=numba.typed.List(init_state.split(";")),
                num_wildcards=num_wild,
            )
        except Exception as e:
            print("creating puzzle failed for", sol_id)
            print("allowed_moves", allowed_moves_nb)
            print("solution_state", sol_state.split(";"))
            print("initial_state", init_state.split(";"))
            print("num_wildcards", num_wild)
            raise e

        # Score submission row
        total_num_moves += score_puzzle(sol_id, puzzle, sub_moves)

    return total_num_moves


spec = [
    ("puzzle_id", numba.types.unicode_type),
    (
        "allowed_moves",
        numba.types.DictType(
            numba.types.unicode_type, numba.types.ListType(numba.types.int64)
        ),
    ),
    ("solution_state", numba.types.ListType(numba.types.unicode_type)),
    ("initial_state", numba.types.ListType(numba.types.unicode_type)),
    ("num_wildcards", numba.types.int64),
]


@numba.experimental.jitclass(spec)
class Puzzle:
    puzzle_id: numba.types.unicode_type
    allowed_moves: numba.typed.Dict[
        numba.types.unicode_type, numba.types.ListType(numba.types.int64)
    ]
    solution_state: numba.typed.List[numba.types.unicode_type]
    initial_state: numba.typed.List[numba.types.unicode_type]
    num_wildcards: numba.types.int64

    def __init__(
        self, puzzle_id, allowed_moves, solution_state, initial_state, num_wildcards
    ):
        self.puzzle_id = puzzle_id
        self.allowed_moves = allowed_moves
        self.solution_state = solution_state
        self.initial_state = initial_state
        self.num_wildcards = num_wildcards


@numba.njit
def check_moves_solve_puzzle(
    puzzle_solution_state: numba.types.ListType(numba.types.int64),
    state: numba.types.ListType(numba.types.int64),
    num_wildcards: numba.types.int64,
):
    num_wrong_facelets = 0
    for s, t in zip(puzzle_solution_state, state):
        if not (s == t):
            num_wrong_facelets += 1
    if num_wrong_facelets > num_wildcards:
        return False
    return True


'''
def apply_move(self, move, inverse=False):
    """
    Apply a move or its inverse to the puzzle state.

    :param move: List representing the move as a permutation.
    :param inverse: Boolean indicating whether to apply the inverse of the move.
    """
    if inverse:
        inverse_move = self.inverse_move(move)
        self.state = [self.state[inverse_move[i]] for i in range(len(self.state))]
    else:
        self.state = [self.state[i] for i in move]

    def inverse_move(self, move):
        return {v: k for k, v in enumerate(move)}
'''


@numba.njit
def apply_move(
    state: numba.types.ListType(numba.types.unicode_type),
    move: numba.types.ListType(numba.types.types.int64),
    inverse_move: numba.types.int64,
):
    if inverse_move:
        inverse_move = {v: k for k, v in enumerate(move)}
        return [state[inverse_move[i]] for i in range(len(state))]
    return [state[i] for i in move]


@numba.njit
def score_puzzle(
    puzzle_id: numba.types.unicode_type,
    puzzle: Puzzle,
    sub_solution: numba.types.unicode_type,
):
    """Score the solution to a permutation puzzle."""
    # Apply submitted sequence of moves to the initial state, from left to right
    moves = sub_solution.split(".")
    state = puzzle.initial_state
    # for m in moves:
    #    power = 1
    #    if m[0] == "-":
    #        m = m[1:]
    #        power = -1
    #    try:
    #        p = puzzle.allowed_moves[m]
    #    except Exception:
    #        raise Exception(f"Move {m} is not allowed for puzzle {puzzle_id}")
    #
    #    state = apply_move(state, p, power)

    check_moves_solve_puzzle(
        puzzle_solution_state=puzzle.solution_state,
        state=state,
        num_wildcards=puzzle.num_wildcards,
    )
    return len(moves)


def main(submission_file_path):
    IN_KAGGLE = False

    kaggle_folder = "/kaggle/input/"
    local_folder = "./data/"
    puzzle_info_path = (
        kaggle_folder if IN_KAGGLE else local_folder + "santa-2023/puzzle_info.csv"
    )
    puzzles_df = pl.read_csv(
        kaggle_folder if IN_KAGGLE else local_folder + "santa-2023/puzzles.csv"
    )
    submission = pl.read_csv(submission_file_path)

    score_ = score(
        solution=puzzles_df,
        submission=submission,
        series_id_column_name="id",
        moves_column_name="moves",
        puzzle_info_path=puzzle_info_path,
    )
    return score_


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        files = ["submission.csv"]
    elif len(args) == 1:
        files = [args[0]]
    else:
        files = args

    for file in args:
        t1 = time.time()
        score_ = main(file)
        t2 = time.time()
        print(
            f"Score for solution file {file}: {score_}, calculated in {t2-t1:.2f} seconds."
        )

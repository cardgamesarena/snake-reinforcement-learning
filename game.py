import numpy as np
from collections import deque
import random

GRID_SIZE = 10
EMPTY = 0
BODY  = 1
HEAD  = 2
FOOD  = 3

# Actions as (row_delta, col_delta)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3

# Opposite actions (to prevent reversing direction)
OPPOSITE = {
    ACTION_UP:    ACTION_DOWN,
    ACTION_DOWN:  ACTION_UP,
    ACTION_LEFT:  ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}


class SnakeGame:
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.max_steps = grid_size * grid_size * 4
        self.snake: deque = deque()
        self.direction: int = ACTION_RIGHT
        self.food: tuple = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.reset()

    def reset(self) -> np.ndarray:
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid, mid - 1)])
        self.direction = ACTION_RIGHT
        self.score = 0
        self.steps = 0
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        snake_set = set(self.snake)
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in snake_set
        ]
        if empty:
            self.food = random.choice(empty)

    def _get_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for r, c in list(self.snake)[1:]:
            grid[r, c] = BODY
        head_r, head_c = self.snake[0]
        grid[head_r, head_c] = HEAD
        food_r, food_c = self.food
        grid[food_r, food_c] = FOOD
        return grid

    def _direction_onehot(self) -> np.ndarray:
        oh = np.zeros(4, dtype=np.float32)
        oh[self.direction] = 1.0
        return oh

    def get_state(self) -> np.ndarray:
        grid = self._get_grid()
        flat = grid.flatten() / 3.0  # normalize to [0, 1]
        direction = self._direction_onehot()
        return np.concatenate([flat, direction])

    def get_score(self) -> int:
        return self.score

    def get_grid_dict(self) -> dict:
        grid = self._get_grid()
        return {
            "grid": grid.astype(int).tolist(),
            "score": self.score,
            "steps": self.steps,
        }

    def step(self, action: int) -> tuple:
        # Prevent 180-degree reversal: ignore action if it would reverse direction
        if action == OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action

        dr, dc = ACTIONS[action]
        head_r, head_c = self.snake[0]
        new_r, new_c = head_r + dr, head_c + dc

        # Wall collision
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            return self.get_state(), -100.0, True

        # Self collision (check against body excluding tail, which will move)
        body_without_tail = set(list(self.snake)[:-1])
        if (new_r, new_c) in body_without_tail:
            return self.get_state(), -100.0, True

        # Move snake
        self.snake.appendleft((new_r, new_c))

        ate_food = (new_r, new_c) == self.food
        if ate_food:
            self.score += 1
            self._place_food()
            reward = 100.0
        else:
            self.snake.pop()  # remove tail
            reward = 0.0

        self.steps += 1

        # Step limit
        if self.steps >= self.max_steps:
            return self.get_state(), -100.0, True

        return self.get_state(), reward, False

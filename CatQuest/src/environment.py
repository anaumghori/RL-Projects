import numpy as np
from typing import Tuple, Dict, Any
import sys
import os
from configs.config import *


class CatPuzzleEnv:
    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.max_steps = MAX_STEPS
        self.broom_pos = BROOM_POS
        self.bathtub_pos = BATHTUB_POS
        self.meat_pos = MEAT_POS
        self.cat_pos = None
        self.step_count = 0
        self.done = False
        self.reset()
    
    # Reset environment to initial state
    def reset(self) -> np.ndarray:
        self.cat_pos = CAT_START_POS
        self.step_count = 0
        self.done = False
        return self._get_state()
    
    # Execute one step in the environment
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        # Get new position after action
        new_pos = self._get_new_position(action)
        
        # Check if move is valid
        if not self._is_valid_move(new_pos):
            reward = INVALID_MOVE_REWARD
            info = {"invalid_move": True}
        else:
            # Execute valid move
            self.cat_pos = new_pos
            reward = self._calculate_reward(new_pos)
            info = {"invalid_move": False}
        
        # Update step count
        self.step_count += 1
        
        # Check termination conditions
        self._check_termination()
        
        # Add timeout penalty if max steps reached
        if self.step_count >= self.max_steps and not self.done:
            reward += TIMEOUT_REWARD
            self.done = True
            info["timeout"] = True
        
        return self._get_state(), reward, self.done, info
    
    # Get current state as multi-channel representation
    def _get_state(self) -> np.ndarray:
        # Create 4-channel state: [cat, meat, bathtub, broom]
        state = np.zeros((self.height, self.width, 4), dtype=np.float32)
        state[self.cat_pos[0], self.cat_pos[1], 0] = 1.0
        state[self.meat_pos[0], self.meat_pos[1], 1] = 1.0
        state[self.bathtub_pos[0], self.bathtub_pos[1], 2] = 1.0
        state[self.broom_pos[0], self.broom_pos[1], 3] = 1.0
        return state
    
    # Calculate new position based on action
    def _get_new_position(self, action: int) -> Tuple[int, int]:
        if action not in ACTIONS:
            return self.cat_pos
        
        delta_row, delta_col = ACTIONS[action]
        new_row = self.cat_pos[0] + delta_row
        new_col = self.cat_pos[1] + delta_col
        
        return (new_row, new_col)
    
    # Check if move is within bounds
    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width
    
    # Calculate reward based on new position
    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        if pos == self.meat_pos:
            return MEAT_REWARD
        elif pos == self.bathtub_pos:
            return BATHTUB_REWARD
        elif pos == self.broom_pos:
            return BROOM_REWARD
        else:
            return STEP_REWARD
    
    # Check if episode should terminate
    def _check_termination(self):
        if self.cat_pos == self.meat_pos:
            self.done = True
    
    def get_action_space_size(self) -> int:
        return len(ACTIONS)
    
    def get_state_shape(self) -> Tuple[int, int, int]:
        return (self.height, self.width, 4)
    
    # Get flattened state for DQN input
    def get_state_flat(self) -> np.ndarray:
        return self._get_state().flatten()
    
    # Get flattened state size
    def get_state_size(self) -> int:
        return self.height * self.width * 4
    
    # Render current state for visualization
    def render(self, mode='human'):
        if mode == 'human':
            # Create visual grid
            grid = np.full((self.height, self.width), EMPTY, dtype=int)
            
            # Place objects, cat gets overwrites if already on same position
            grid[self.bathtub_pos[0], self.bathtub_pos[1]] = BATHTUB
            grid[self.broom_pos[0], self.broom_pos[1]] = BROOM
            grid[self.meat_pos[0], self.meat_pos[1]] = MEAT
            grid[self.cat_pos[0], self.cat_pos[1]] = CAT
            
            # Print grid
            print(f"Step: {self.step_count}")
            for row in range(self.height):
                for col in range(self.width):
                    print(SYMBOLS[grid[row, col]], end=' ')
                print()
            print()
    
    # Check if current position is goal
    def is_goal_reached(self) -> bool:
        return self.cat_pos == self.meat_pos
    
    def get_cat_position(self) -> Tuple[int, int]:
        return self.cat_pos
    
    def get_remaining_steps(self) -> int:
        return max(0, self.max_steps - self.step_count)
    
    # Convert 2D position to 1D index
    def _pos_to_index(self, row: int, col: int) -> int:
        return row * self.width + col
    
    # Convert 1D index to 2D position
    def _index_to_pos(self, index: int) -> Tuple[int, int]:
        return index // self.width, index % self.width
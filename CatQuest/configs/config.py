GRID_HEIGHT = 4
GRID_WIDTH = 3
MAX_STEPS = 15

# Fixed positions
CAT_START_POS = (0, 0)
BROOM_POS = (1, 2)
BATHTUB_POS = (2, 0)
MEAT_POS = (3, 1)

# Rewards
STEP_REWARD = -0.1
BROOM_REWARD = -2.0
BATHTUB_REWARD = -5.0
MEAT_REWARD = 10.0
INVALID_MOVE_REWARD = -1.0
TIMEOUT_REWARD = -1.0

# Action mappings
ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

ACTION_NAMES = {
    0: "UP",
    1: "DOWN", 
    2: "LEFT",
    3: "RIGHT"
}

# Symbolic encoding for grid elements
EMPTY = 0
CAT = 1
MEAT = 2
BATHTUB = 3
BROOM = 4

# Grid element symbols for visualization
SYMBOLS = {
    EMPTY: "⬜",
    CAT: "🐱",
    MEAT: "🥩",
    BATHTUB: "🛁",
    BROOM: "🧹"
}

# DQN hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
HIDDEN_SIZE = 128
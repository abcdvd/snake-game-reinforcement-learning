import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # exploration rate
        self.gamma = 0.9    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # experience replay memory
        # Model with updated input size (30 inputs) for expanded state representation
        self.model = Linear_QNet(30, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]  # snake's head position
        # Points one step ahead in each direction from the head
        point_l  = Point(head.x - BLOCK_SIZE, head.y)
        point_r  = Point(head.x + BLOCK_SIZE, head.y)
        point_u  = Point(head.x, head.y - BLOCK_SIZE)
        point_d  = Point(head.x, head.y + BLOCK_SIZE)
        # Points two steps ahead in each direction from the head
        point_l2 = Point(head.x - 2 * BLOCK_SIZE, head.y)
        point_r2 = Point(head.x + 2 * BLOCK_SIZE, head.y)
        point_u2 = Point(head.x, head.y - 2 * BLOCK_SIZE)
        point_d2 = Point(head.x, head.y + 2 * BLOCK_SIZE)
        # Points three steps ahead in each direction from the head
        point_l3 = Point(head.x - 3 * BLOCK_SIZE, head.y)
        point_r3 = Point(head.x + 3 * BLOCK_SIZE, head.y)
        point_u3 = Point(head.x, head.y - 3 * BLOCK_SIZE)
        point_d3 = Point(head.x, head.y + 3 * BLOCK_SIZE)
        # Diagonal neighboring points (one step away)
        point_ul = Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_ur = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_dl = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
        point_dr = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)

        # Current direction booleans
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Distances to walls (in blocks) from current head position
        dist_left_wall  = head.x // BLOCK_SIZE
        dist_right_wall = (game.w // BLOCK_SIZE) - 1 - (head.x // BLOCK_SIZE)
        dist_up_wall    = head.y // BLOCK_SIZE
        dist_down_wall  = (game.h // BLOCK_SIZE) - 1 - (head.y // BLOCK_SIZE)

        # Construct the state with expanded awareness
        state = [
            # Danger straight (1, 2, and 3 steps ahead)
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),                     # 1 step ahead
            (dir_r and game.is_collision(point_r2)) or 
            (dir_l and game.is_collision(point_l2)) or 
            (dir_u and game.is_collision(point_u2)) or 
            (dir_d and game.is_collision(point_d2)),                    # 2 steps ahead
            (dir_r and game.is_collision(point_r3)) or 
            (dir_l and game.is_collision(point_l3)) or 
            (dir_u and game.is_collision(point_u3)) or 
            (dir_d and game.is_collision(point_d3)),                    # 3 steps ahead

            # Danger right (1, 2, and 3 steps ahead)
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),                     # 1 step (to the right of current direction)
            (dir_u and game.is_collision(point_r2)) or 
            (dir_d and game.is_collision(point_l2)) or 
            (dir_l and game.is_collision(point_u2)) or 
            (dir_r and game.is_collision(point_d2)),                    # 2 steps (to the right)
            (dir_u and game.is_collision(point_r3)) or 
            (dir_d and game.is_collision(point_l3)) or 
            (dir_l and game.is_collision(point_u3)) or 
            (dir_r and game.is_collision(point_d3)),                    # 3 steps (to the right)

            # Danger left (1, 2, and 3 steps ahead)
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),                     # 1 step (to the left of current direction)
            (dir_d and game.is_collision(point_r2)) or 
            (dir_u and game.is_collision(point_l2)) or 
            (dir_r and game.is_collision(point_u2)) or 
            (dir_l and game.is_collision(point_d2)),                    # 2 steps (to the left)
            (dir_d and game.is_collision(point_r3)) or 
            (dir_u and game.is_collision(point_l3)) or 
            (dir_r and game.is_collision(point_u3)) or 
            (dir_l and game.is_collision(point_d3)),                    # 3 steps (to the left)

            # Move direction (one-hot encoding of current direction)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location (relative to head)
            game.food.x < game.head.x,  # food is to the left
            game.food.x > game.head.x,  # food is to the right
            game.food.y < game.head.y,  # food is above
            game.food.y > game.head.y,  # food is below

            # Surrounding cell obstacles (immediate grid around head)
            game.is_collision(point_ul),  # obstacle at up-left
            game.is_collision(point_u),   # obstacle directly up
            game.is_collision(point_ur),  # obstacle at up-right
            game.is_collision(point_l),   # obstacle directly left
            game.is_collision(point_r),   # obstacle directly right
            game.is_collision(point_dl),  # obstacle at down-left
            game.is_collision(point_d),   # obstacle directly down
            game.is_collision(point_dr),  # obstacle at down-right

            # Distances to walls (up, down, left, right)
            dist_left_wall,   # distance to left wall
            dist_right_wall,  # distance to right wall
            dist_up_wall,     # distance to top wall
            dist_down_wall,   # distance to bottom wall

            # Current snake length
            len(game.snake)
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train on a batch of experiences from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # random batch
        else:
            mini_sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on a single step (for sequential learning)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Determine action based on exploration-exploitation tradeoff
        self.epsilon = 80 - self.n_games  # higher epsilon at start for more exploration
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # Exploration: random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: choose best move from model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move (based on current state)
        final_move = agent.get_action(state_old)

        # Perform move and get reward
        reward, done, score = game.play_step(final_move)
        # Get new state after move
        state_new = agent.get_state(game)

        # Train short memory for the move
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # Remember the experience
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game over: reset environment and train long memory on the episode
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Track record score and print results
            if score > record:
                record = score
                agent.model.save()  # save model if new record achieved
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')

            # Plot scores
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
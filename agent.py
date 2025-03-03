import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import CNN_QNet, QTrainer
from helper import plot, plot_loss

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # grid: 24x32 (640/20 x 480/20), plus 11 features, 3 actions
        self.model = CNN_QNet((24, 32), 11, 3).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        grid_state = []
        block_size = 20  # or from game.py

        # fill the grid info
        for y in range(0, game.h, block_size):
            for x in range(0, game.w, block_size):
                point = Point(x, y)
                if point == game.snake[0]:
                    grid_state.append(5)
                elif point in game.snake:
                    grid_state.append(1)
                elif game.food == point:
                    grid_state.append(10)   # a different marker for food
                else:
                    grid_state.append(0)

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        grid_state.extend(state)
        return np.array(grid_state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss
    
    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    def get_action(self, state):
        self.epsilon = max(10, 150 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            if len(state0.shape) == 1:
                state0 = state0.unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction, dim=1).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    plot_mean_losses = []

    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory & accumulate the loss
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # game over => train long memory
            game.reset()
            agent.n_games += 1
            loss = agent.train_long_memory()

            # update record if needed
            if score > record:
                record = score
                agent.model.save()

            # print result
            print("Game", agent.n_games, "Score", score, "Record", record, "Loss", round(loss, 4))

            # track the score
            plot_scores.append(score)
            mean_score = sum(plot_scores[-10:]) / min(10, agent.n_games)
            plot_mean_scores.append(mean_score)

            # track the final loss of this episode (you can also track average episode loss if you want)
            plot_losses.append(loss)
            mean_loss = sum(plot_losses[-10:]) / min(10, agent.n_games)
            plot_mean_losses.append(mean_loss)

            # update plots
            plot(plot_scores, plot_mean_scores, plot_mean_losses)
            # plot_loss(plot_losses, plot_mean_losses)


if __name__ == '__main__':
    train()

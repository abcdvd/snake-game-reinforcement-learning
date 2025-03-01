import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Initialize pygame and font
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Define game constants
BLOCK_SIZE = 20
SPEED = 2000

# Define directions for snake movement
class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

# Point on the grid
Point = namedtuple('Point', 'x, y')

# RGB colors for display
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Initialize display for the game
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Reset game state for a new episode
        self.direction = Direction.RIGHT
        # Starting snake (length 3)
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0  # to track game loops (to prevent infinite loops)

    def _place_food(self):
        # Place food at a random position not occupied by the snake
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            # If food spawned on the snake, place it again
            self._place_food()

    def play_step(self, action):
        # Handle events (allows closing the game window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.frame_iteration += 1
        # Move the snake based on the action (action is [straight, right, left])
        self._move(action)
        # Insert new head position
        self.snake.insert(0, self.head)

        # Check if game over (collision or stuck in loop)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # Move snake: remove tail segment
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x < 0 or pt.x > self.w - BLOCK_SIZE or pt.y < 0 or pt.y > self.h - BLOCK_SIZE:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Draw background
        self.display.fill(BLACK)
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        # Refresh display
        pygame.display.flip()

    def _move(self, action):
        # Map the action (straight, right, left) to a new direction
        # Possible moves relative to current direction
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # No change, go straight
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Right turn relative to current direction
            new_dir = clockwise[(idx + 1) % 4]
        else:  # action == [0, 0, 1]
            # Left turn relative to current direction
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        # Update the head position based on new direction
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        # New head
        self.head = Point(x, y)
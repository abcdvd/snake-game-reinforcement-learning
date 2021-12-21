# snake-game-reinforcement-learning
In 2016 AlphaGo developed by Deepmind corporation play a game with world finest player Lee Se-dol and won 4 by 1. And the next model that deepmind had built is alphastar which plays starcraft 2 and this time it won european champion in 5 by 0. Now a day deepmind successfuly predicted protein folding with accuracy 99.09%, one of the toughest problems in science. This three large achievements is basesd on reinforcement learning algorithm that using deep learning technique from experience data. I'm very shocked and extremely attracted by this result so i want to study some simple reinforcement learning by implementing snake game. The each snake block is 20px and game size is width=640px, height=480px.

![Best](https://user-images.githubusercontent.com/87563747/146955117-a1d34a12-4451-46c4-a414-8a883be99dd8.gif)    
_captured in game_

# Study

![700 game](https://user-images.githubusercontent.com/87563747/146956643-dd3497a5-3c14-4636-bcb6-320eb3eb89ab.png)
<img src="https://user-images.githubusercontent.com/87563747/146956643-dd3497a5-3c14-4636-bcb6-320eb3eb89ab.png" width="500" height="300"/>
          
Up figure is original code result. As i can see agent struggle under 80 number of games then it explod. Because if `n_games` is under 80, then `self.epsilon` is positive so the random method will not work instead learning algorithm will work.

```python
def get_action(self, state):
    self.epsilon = 80 - self.n_games
    final_move = [0,0,0]
    if random.randint(0, 200) < self.epsilon:
        move = random.randint(0, 2)
        final_move[move] = 1
    else:
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        
    return final_move
```

But after `n_games > 80` average score is converging to some number. Because i could observe snake being trapped on their own without continuing to eat. To overcome this problem i should try a few things.

## Changing reward policy

+ Current reward system

    The current reward system is when snake eat food then reward 10 is given and when game is over or time is exceeded then reward is -10. Check code below.
    ```python
    if self.is_collision() or self.frame_iteration > 100*len(self.snake):
    game_over = True
    reward = -10
    return reward, game_over, self.scor
    
    if self.head == self.food:
        self.score += 1
        reward = 10
        self._place_food()
    else:
        self.snake.pop()
    ```
    I change the value of reward ,but it didn't change anything.
    


    

# Reference
I watch the YouTube Video "https://youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV" and his Github code "https://github.com/python-engineer/snake-ai-pytorch"

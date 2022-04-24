# snake-game-reinforcement-learning
In 2016 AlphaGo developed by Deepmind corporation play a game with world finest player Lee Se-dol and won 4 by 1. And the next model that deepmind had built is alphastar which plays starcraft 2 and this time it won european champion in 5 by 0. Now a day deepmind successfuly predicted protein folding with accuracy 99.09%, one of the toughest problems in science. This three large achievements is basesd on reinforcement learning algorithm that using deep learning technique from experience data. I'm very shocked and extremely attracted by this result so i want to study some simple reinforcement learning by implementing snake game. The each snake block is 20px and game size is width=640px, height=480px.

![Best](https://user-images.githubusercontent.com/87563747/146955117-a1d34a12-4451-46c4-a414-8a883be99dd8.gif)    
_captured in game_

# Study

![700 game](https://user-images.githubusercontent.com/87563747/146956643-dd3497a5-3c14-4636-bcb6-320eb3eb89ab.png)
          
Up figure is original code result which is the overall learning curve. As you can see agent struggle under 80 number of games then after that it exploded. Because agent needed data to learn, so I set epsilon that give enough chance to agent to collect data. 

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
if `n_games` is under 80, then `self.epsilon` is positive so lower `n_games` have bigger probability that random decision method will work instead of reinforcement learning algorithm. In the other way if `n_games` is upper 80, then `self.epsilon`is negative which mean that algorithm always decide the agent movements.

But after `n_games > 80` average score (orange line) is converging to some number. Because i could observe snake being trapped on their own without continuing to eat. To overcome this problem i should try a few things.

## Problem Solving

+ Changing reward system

    The current reward system activate like when snake eat food then reward 10 is given and when game is over or time is exceeded then reward is -10. Check code below.
    
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
    So I wonder if the results will change if the amount of reward changes. I design `for loop` to check every `mean_score` in range `10<=plus_reward<=20`, `-20<=minus_reward<=-10` about 300 games iteration.
    ```python
    if __name__ == '__main__':
          plus_list = [i+10 for i in range(11)]
          minus_list= [-i-10 for i in range(11)]
          list_data = []

          for plus in plus_list:
              atom_list=[]
              for minus in minus_list:
                  mean = train(minus, plus)
                  atom_list.append(mean)
              list_data.append(atom_list)
    
    df = pd.DataFrame(list_data, index=plus_list, columns=minus_list)
    print(df)
    df.to_csv("11x11_research.csv")
    ```
    | **plus_reward, minus_reward** | **-10** | **-11** | **-12** | **-13** | **-14** | **-15** | **-16** | **-17** | **-18** | **-19** | **-20** |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **10** | 19.69 | 15.48 | 24.66 | 21.15 | 23.57 | 24.58 | 20.52 | 23.91 | 23.68 | 22.32 | 18.47 |
    | **11** | 22.02 | 20.87 | 24.31 | 23.21 | 24.06 | 23.88 | 16.97 | 22.3 | 18.76 | 20.89 | 20.4 |
    | **12** | 23.45 | 23.2 | 21.27 | 22.25 | 21.42 | 24.24 | 23.94 | 23.78 | 23.85 | 18.85 | 21.02 |
    | **13** | 24.35 | 22.94 | 23.09 | 21.77 | 24.64 | 21.8 | 22.86 | 22.88 | 23.61 | 20.7 | 23.51 |
    | **14** | 22.54 | 23.48 | 21.52 | 24.87 | 22.57 | 21.2 | 23.63 | 24.31 | 20.97 | 20.57 | 18.0 |
    | **15** | 22.61 | 22.83 | 21.7 | 22.23 | 20.64 | 16.55 | 21.8 | 24.56 | 22.55 | 21.73 | 23.11 |
    | **16** | 23.05 | 21.45 | 22.12 | 22.27 | 20.57 | 23.35 | 24.11 | 22.38 | 23.84 | 21.67 | 23.23 |
    | **17** | 22.42 | 22.34 | 22.53 | 23.26 | 23.33 | 23.92 | 24.21 | 22.46 | 22.44 | 23.4 | 24.73 |
    | **18** | 20.29 | 22.19 | 20.4 | 19.93 | 21.8 | 21.66 | 22.82 | 23.4 | 20.32 | 15.12 | 23.45 |
    | **19** | 22.33 | 23.05 | 21.93 | 23.4 | 22.36 | 23.81 | 21.55 | 17.06 | 23.24 | 21.99 | 18.2 |
    | **20** | 22.15 | 21.55 | 22.25 | 22.53 | 19.75 | 22.63 | 18.88 | 22.55 | 22.29 | 19.84 | 23.05 |
    
    
    Changing Value doesn't make meanignful difference.
    

+ Adding more data of condition to Snake

  The main problem of snake is that it keeps locking itself. To alert this danger to snake additional imformation must be needed.

  + Location of last tail

# Reference
I watch the YouTube Video "https://youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV" and his Github code "https://github.com/python-engineer/snake-ai-pytorch"

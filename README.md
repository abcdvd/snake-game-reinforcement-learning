# snake-game-reinforcement-learning
In 2016 AlphaGo developed by Deepmind corporation play a game with world finest player Lee Se-dol and won 4 by 1. And the next model that deepmind had built is alphastar which plays starcraft 2 and this time it won european champion in 5 by 0. Now a day deepmind successfuly predicted protein folding with accuracy 99.09%, one of the toughest problems in science. This three large achievements is basesd on reinforcement learning algorithm that using deep learning technique from experience data. I'm very shocked and extremely attracted by this result so i want to study some simple reinforcement learning by implementing snake game. The each snake block is 20px and game size is width=640px, hight=480px.
# Reference
I watch the YouTube Video "https://youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV" and his Github code "https://github.com/python-engineer/snake-ai-pytorch"
# Study

![orginal-reward-system](https://user-images.githubusercontent.com/87563747/146942688-30906709-f262-45db-9e02-2af55a6c49a4.png)

Up figure is original code result. As i can see agent struggle under 80 number of games then it exploed.

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

The average score is converging to some number. Because snake get trapped itself.

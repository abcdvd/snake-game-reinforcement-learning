import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to torch tensors
        state      = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32), dtype=torch.float)
        action     = torch.tensor(np.array(action, dtype=np.int64))
        reward     = torch.tensor(np.array(reward, dtype=np.float32), dtype=torch.float)

        # If only a single sample, add batch dimension
        if len(state.shape) == 1:
            state      = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action, 0)
            reward     = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q values for current state
        pred = self.model(state)
        # Clone predictions to compute target values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Bellman equation for Q-value update
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # Update the Q-value for the action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
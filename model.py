import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device=None):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        # 디바이스 설정: 명시적으로 전달되지 않으면 자동으로 할당
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 모델을 지정한 디바이스로 이동

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, device=None):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # 디바이스 설정: 모델에 설정된 디바이스를 사용하거나, 명시적으로 전달한 디바이스를 사용
        self.device = device if device is not None else (self.model.device if hasattr(self.model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 모든 입력 데이터를 텐서로 변환하고, 지정한 디바이스로 이동
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        
        # 만약 입력 데이터가 배치 차원이 없다면 추가 (1, x) 형태로 변경
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = torch.tensor([done], dtype=torch.bool).to(self.device)
        else:
            done = torch.tensor(done, dtype=torch.bool).to(self.device)

        # 현재 상태를 바탕으로 예측한 Q값
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # next_state[idx]는 1차원 텐서이므로, unsqueeze(0)로 배치 차원을 추가해 모델에 전달
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))[0]
            # 행동의 인덱스에 해당하는 Q값을 갱신
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
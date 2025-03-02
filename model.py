import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class CNN_QNet(nn.Module):
    def __init__(self, grid_shape, additional_features, output_size, device=None):
        """
        grid_shape: (rows, cols) 예를 들어 (24, 32)
        additional_features: 추가 피처 개수 (예: 11)
        output_size: 행동의 수 (예: 3)
        """
        super(CNN_QNet, self).__init__()
        self.grid_shape = grid_shape
        self.additional_features = additional_features
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CNN 부분: grid 입력 (1 채널, 24x32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 출력: (16, 12, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 출력: (32, 6, 8)
        
        # flattened conv output 크기 계산: 32 * (rows/4) * (cols/4)
        self.flattened_size = 32 * (self.grid_shape[0] // 4) * (self.grid_shape[1] // 4)
        
        # CNN 출력을 추가 피처와 합친 후의 fully connected layer
        self.fc1 = nn.Linear(self.flattened_size + additional_features, 512)
        self.fc2 = nn.Linear(512, output_size)
        
        self.to(self.device)  # 모델 전체를 device로 이동

    def forward(self, x):
        """
        입력 x: shape (batch, total_features)
          total_features = grid_flat (grid_shape[0]*grid_shape[1]) + additional_features
        """
        batch_size = x.shape[0]
        grid_size = self.grid_shape[0] * self.grid_shape[1]
        # grid 부분과 추가 피처 분리
        grid = x[:, :grid_size]
        extra = x[:, grid_size:]
        # grid를 (batch, 1, rows, cols)로 reshape
        grid = grid.view(batch_size, 1, self.grid_shape[0], self.grid_shape[1])
        
        # CNN 처리
        grid = F.relu(self.conv1(grid))
        grid = self.pool1(grid)
        grid = F.relu(self.conv2(grid))
        grid = self.pool2(grid)
        grid = grid.view(batch_size, -1)  # flatten
        
        # CNN 결과와 추가 피처를 concat
        x = torch.cat((grid, extra), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        self.device = device if device is not None else (self.model.device if hasattr(self.model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 모든 입력 데이터를 텐서로 변환하고 지정한 device로 이동
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = torch.tensor([done], dtype=torch.bool).to(self.device)
        else:
            done = torch.tensor(done, dtype=torch.bool).to(self.device)

        # 현재 상태에 대한 Q값 예측
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0))).item()
            # 해당 행동 인덱스에 Q값 갱신
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
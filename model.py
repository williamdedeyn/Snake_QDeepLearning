import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        path = "./models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self,file_name="model.pth"):
        path = "./models"
        if not os.path.exists(path):
           print("Path does not exist")
        file_name = os.path.join(path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()



class Qtrainer:

    def __init__(self,lr,gamma,model):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def predict(self,state):
        state = torch.tensor(state, dtype=torch.float)
        return self.model(state)

    def train_step(self,state,next_state,action,reward,game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)


        if len(state.shape) == 1:
            # reshaping the state to 2D array
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over,)

            pred = self.model(state)
            target = pred.clone()
            for idx in range(len(game_over)):
                Q_new = reward[idx]

                if not game_over[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(
                        self.model(next_state[idx])
                    )

                target[idx][torch.argmax(action).item()] = Q_new

                # 4. compute loss
            self.optimizer.zero_grad()  # empty the gradients
            loss = self.criterion(target, pred)

            # 5. backpropagate
            loss.backward()  # backpropagation

            # 6. update the weights
            self.optimizer.step()  # update the weights


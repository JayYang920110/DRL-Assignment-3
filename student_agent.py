import torch
from torch import nn
import random
import numpy as np
import cv2
from collections import deque
# QNet
class QNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNet, self).__init__()
        self.input_shape = input_shape
        self.num_actions = n_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self._feature_size = self._get_feature_size()

        self.fc_value = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def _get_feature_size(self):
        with torch.no_grad():
            x = self.features(torch.zeros(1, *self.input_shape))
            return x.view(1, -1).size(1)

# DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size, device, gamma=0.99, lr=1e-4, buffer_size=100000, batch_size=128, target_update_freq=1000, init_learning=20000, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.qnet_local = QNet(state_size, action_size).to(self.device)
        self.qnet_target = QNet(state_size, action_size).to(self.device)

# Agent

class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.agent = DQNAgent((4, 84, 84), 12, device=self.device)

        checkpoint = torch.load("./model.pth", map_location=self.device)
        self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])

        self.raw_obs_buffer = deque(maxlen=2)  # 模仿 MaxAndSkipEnv 的兩幀 buffer
        self.stack_buffer = np.zeros((4, 84, 84), dtype=np.float32)  # frame stack
        self.last_action = 0
        self.frame_count = 0  # 幀計數

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        resized = resized[:,:,None]
        state = np.moveaxis(resized, 2, 0)
        return state # shape: (1, 84, 84)
    def act(self, observation):

        self.raw_obs_buffer.append(observation)

        if self.frame_count == 0:
            self.raw_obs_buffer.clear()
            self.stack_buffer[:] = 0

        if self.frame_count % 4 == 0:
            self.frame_count += 1
            max_frame = np.max(np.stack(self.raw_obs_buffer), axis=0)
            obs = self.preprocess(max_frame)
            self.stack_buffer[:-1] = self.stack_buffer[1:]
            self.stack_buffer[-1] = obs
            norm_stack = self.stack_buffer.astype(np.float32) / 255.0
            self.last_action = self.select_action(norm_stack)
            return self.last_action
        
        else:
            self.frame_count += 1
            return self.last_action
    def select_action(self, stack):
        eps = 0.01  # inference 固定低 epsilon
        if random.random() < eps:
            return random.randint(0, self.agent.action_size - 1)
        else:
            state_tensor = torch.from_numpy(stack).unsqueeze(0).to(self.device)  # shape: (1, 4, 84, 84)
            self.agent.qnet_local.eval()
            with torch.no_grad():
                q_values = self.agent.qnet_local(state_tensor)
            return q_values.argmax(1).item()
    # def act(self, observation):
    #     self.frame_count += 1

    #     if self.frame_count == 1:
    #         self.raw_obs_buffer.clear()
    #         self.stack_buffer[:] = 0

    #         self.raw_obs_buffer.append(observation)

    #         obs = self.preprocess(self.raw_obs_buffer[-1])
    #         self.stack_buffer[:-1] = self.stack_buffer[1:]
    #         self.stack_buffer[-1] = obs
    #         norm_stack = self.stack_buffer.astype(np.float32) / 255.0
    #         self.last_action = self.select_action(norm_stack)
    #         return self.last_action  
    #     # 儲存最新 obs，模仿 MaxAndSkipEnv (保留兩幀)
    #     if self.frame_count % 4 == 1:
    #         self.raw_obs_buffer.append(observation)

    #         # obs = self.preprocess(self.raw_obs_buffer[-1])
    #         # self.stack_buffer[:-1] = self.stack_buffer[1:]
    #         # self.stack_buffer[-1] = obs
            
    #         self.last_action = self.select_action(self.stack_buffer)
    #         return self.last_action
    #     else:
    #         if self.frame_count % 4 == 2 or self.frame_count % 4 == 3:
    #             self.raw_obs_buffer.append(observation)
    #             return self.last_action
    #         elif self.frame_count % 4 == 0:
    #             self.raw_obs_buffer.append(observation)
    #             max_frame = np.max(np.stack(self.raw_obs_buffer), axis=0)
    #             obs = self.preprocess(max_frame)
    #             self.stack_buffer[:-1] = self.stack_buffer[1:]
    #             self.stack_buffer[-1] = obs
    #             self.stack_buffer = self.stack_buffer.astype(np.float32) / 255.0
    #             return self.last_action






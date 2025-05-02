import gym
import torch
from torch import nn
import numpy as np
import random
from PIL import Image

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


# Preprocessing
import numpy as np
import cv2
from collections import deque

# Agent
import gym
import numpy as np
import torch
from collections import deque
import cv2

class Agent:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = gym.spaces.Discrete(12)

        self.agent = DQNAgent((4, 84, 84), 12, device=self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])

        self.raw_obs_buffer = deque(maxlen=2)  # 模仿 MaxAndSkipEnv 的兩幀 buffer
        self.stack_buffer = np.zeros((4, 84, 84), dtype=np.float32)  # frame stack
        self.frame_count = 0
        self.eps = 0.01

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=0).astype(np.float32)  # shape: (1, 84, 84)

    def reset(self, initial_obs):
        self.raw_obs_buffer.clear()
        self.frame_count = 0

        preprocessed = self.preprocess(initial_obs)
        self.stack_buffer[:] = np.repeat(preprocessed, 4, axis=0)

    def act(self, obs):
        self.frame_count += 1
        preprocessed = self.preprocess(obs)
        self.raw_obs_buffer.append(preprocessed)

        if self.frame_count % 4 in (2, 3):
            return self.agent.last_action  # 重複動作

        if len(self.raw_obs_buffer) == 2:
            max_frame = np.maximum(self.raw_obs_buffer[0], self.raw_obs_buffer[1])
        else:
            max_frame = self.raw_obs_buffer[0]

        self.stack_buffer[:-1] = self.stack_buffer[1:]
        self.stack_buffer[-1] = max_frame[0]  # squeeze shape (1, 84, 84) → (84, 84)

        input_state = self.stack_buffer / 255.0  # normalize to [0, 1]
        action = self.agent.get_action(input_state, self.eps)
        self.agent.last_action = action
        return action

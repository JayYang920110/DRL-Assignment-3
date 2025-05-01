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
import numpy as np
import torch
import random
import cv2


class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.agent = DQNAgent((4, 84, 84), 12, device=self.device)
        checkpoint = torch.load("./model.pth", map_location=self.device)
        self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])
        self._obs_buffer = deque(maxlen=2)  # 直接用 deque 與 wrapper 一致
        self.buffer = np.zeros((4, 84, 84), dtype=np.float32)  # 與 FrameBuffer 命名一致
        self.last_action = 0
        self.frame_count = 0
        self._skip = 4  # 與 MaxAndSkipEnv 一致

    def reset(self, initial_obs=None):
        self._obs_buffer.clear()
        self.buffer = np.zeros((4, 84, 84), dtype=np.float32)  # 與 FrameBuffer.reset() 一致
        self.last_action = 0
        self.frame_count = 0
        
        # 若有初始 obs，則用它填滿 frame stack（4 張一樣的）
        if initial_obs is not None:
            # 按照 wrapper 順序處理: MaxAndSkipEnv(存一份) → FrameDownsample → ImageToPyTorch → 準備放入 FrameBuffer
            self._obs_buffer.append(initial_obs)
            processed = self._process_single_frame(initial_obs)  # (1, 84, 84)
            # 模擬 FrameBuffer 填滿 4 幀
            for i in range(4):
                self.buffer[i] = processed

    def _process_single_frame(self, obs):
        """處理單幀圖像，按照 wrapper 順序"""
        # 等同於 FrameDownsample
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 等同於 ImageToPyTorch (channel first)
        frame = resized.astype(np.float32)[None, :, :]  # shape: (1, 84, 84)
        
        # 等同於 NormalizeFloats
        return frame / 255.0

    def act(self, observation):
        self.frame_count += 1
        self._obs_buffer.append(observation)
        
        # 達到 skip 數量或是第一次處理時才進行處理
        if self.frame_count % self._skip == 0 or len(self._obs_buffer) == 1:
            # 等同於 MaxAndSkipEnv 的 max_frame
            if len(self._obs_buffer) == 2:
                max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
            else:
                max_frame = self._obs_buffer[0]
            
            # 處理圖像
            processed_frame = self._process_single_frame(max_frame)  # (1, 84, 84)
            
            # 等同於 FrameBuffer 的 observation
            self.buffer[:-1] = self.buffer[1:]  # 向前滾動
            self.buffer[-1] = processed_frame
            
            # 如果已經收集了足夠的幀數（至少4個处理过的幀），则做决策
            if self.frame_count >= self._skip:  # 至少经过一次完整的skip
                self.last_action = self.select_action(self.buffer)
        
        return self.last_action

    def select_action(self, state):
        eps = 0.01  # inference 固定低 epsilon
        if random.random() < eps:
            return random.randint(0, self.agent.action_size - 1)
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # shape: (1, 4, 84, 84)
            self.agent.qnet_local.eval()
            with torch.no_grad():
                q_values = self.agent.qnet_local(state_tensor)
            return q_values.argmax(1).item()

        # # 第4幀來了：模擬 MaxAndSkipEnv (skip=4) 的 max(obs[-2], obs[-1])
        # obs3 = self.preprocess(self.raw_obs_buffer[-2])
        # obs4 = self.preprocess(self.raw_obs_buffer[-1])
        # max_frame = np.maximum(obs3, obs4)

        # # 更新 frame stack
        # self.stack_buffer[:-1] = self.stack_buffer[1:]
        # self.stack_buffer[-1] = max_frame

        
        # self.raw_obs_buffer.clear()

        # # Epsilon-greedy 動作選擇
        # eps = 0.01
        # if random.random() < eps:
        #     self.last_action = random.randint(0, self.agent.action_size - 1)
        # else:
        #     state_tensor = torch.from_numpy(self.stack_buffer).unsqueeze(0).to(self.device)
        #     self.agent.qnet_local.eval()
        #     with torch.no_grad():
        #         q_values = self.agent.qnet_local(state_tensor)
        #     self.last_action = q_values.argmax(1).item()

        # return self.last_action



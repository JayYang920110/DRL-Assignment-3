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

    def get_action(self, state, eps=0.):
        if random.random() > eps:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            self.qnet_local.eval()
            with torch.no_grad():
                action_value = self.qnet_local(state)
            return action_value.max(1)[1].item()
        else:
            return random.choice(np.arange(self.action_size))

# Preprocessing
import numpy as np
import cv2
from collections import deque

# Agent
class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.agent = DQNAgent((4, 84, 84), 12, device=self.device)

        checkpoint = torch.load("./model.pth", map_location=self.device)
        self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])

        self.raw_obs_buffer = []  # 暫存連續4幀原始obs
        self.stack_buffer = np.zeros((4, 84, 84), dtype=np.float32)  # frame stack
        self.last_action = 0
        self.frame_count = 0  # 記錄幾幀進來了

    def reset(self):
        self.raw_obs_buffer.clear()
        self.stack_buffer[:] = 0
        self.last_action = 0
        self.frame_count = 0

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def act(self, observation):
        self.frame_count += 1
        self.raw_obs_buffer.append(observation)

        # 還沒收集滿 4 幀就先回傳上次的動作（cold start）
        if len(self.raw_obs_buffer) < 4:
            return self.last_action
 
        # 第4幀來了：模擬 MaxAndSkipEnv (skip=4) 的 max(obs[-2], obs[-1])
        obs3 = self.preprocess(self.raw_obs_buffer[-2])
        obs4 = self.preprocess(self.raw_obs_buffer[-1])
        max_frame = np.maximum(obs3, obs4)

        # 更新 frame stack
        self.stack_buffer[:-1] = self.stack_buffer[1:]
        self.stack_buffer[-1] = max_frame

        # 清除前4幀中的第一幀，維持最多保留4幀
        if len(self.raw_obs_buffer) > 4:
            self.raw_obs_buffer.clear()

        # Epsilon-greedy 動作選擇
        eps = 0.01
        if random.random() < eps:
            self.last_action = random.randint(0, self.agent.action_size - 1)
        else:
            state_tensor = torch.from_numpy(self.stack_buffer).unsqueeze(0).to(self.device)
            self.agent.qnet_local.eval()
            with torch.no_grad():
                q_values = self.agent.qnet_local(state_tensor)
            self.last_action = q_values.argmax(1).item()

        return self.last_action


# import gym
# import torch
# from torch import nn
# import numpy as np
# import random
# from PIL import Image

# # QNet
# class QNet(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(QNet, self).__init__()
#         self.input_shape = input_shape
#         self.num_actions = n_actions

#         self.features = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         self._feature_size = self._get_feature_size()

#         self.fc_value = nn.Sequential(
#             nn.Linear(self._feature_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)
#         )

#         self.fc_advantage = nn.Sequential(
#             nn.Linear(self._feature_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )

#     def forward(self, x):
#         x = self.features(x).view(x.size(0), -1)
#         value = self.fc_value(x)
#         advantage = self.fc_advantage(x)
#         q = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return q

#     def _get_feature_size(self):
#         with torch.no_grad():
#             x = self.features(torch.zeros(1, *self.input_shape))
#             return x.view(1, -1).size(1)

# # DQNAgent
# class DQNAgent:
#     def __init__(self, state_size, action_size, device, gamma=0.99, lr=1e-4, buffer_size=100000, batch_size=128, target_update_freq=1000, init_learning=20000, tau=1e-3):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.device = device

#         self.qnet_local = QNet(state_size, action_size).to(self.device)
#         self.qnet_target = QNet(state_size, action_size).to(self.device)

#     def get_action(self, state, eps=0.):
#         if random.random() > eps:
#             state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
#             self.qnet_local.eval()
#             with torch.no_grad():
#                 action_value = self.qnet_local(state)
#             return action_value.max(1)[1].item()
#         else:
#             return random.choice(np.arange(self.action_size))

# # Preprocessing
# class FrameDownsampleIdentity:
#     def __init__(self, width=84, height=84):
#         self._width = width
#         self._height = height

#     def __call__(self, observation):
#         img = Image.fromarray(observation)
#         img = img.convert('L')
#         img = img.resize((self._width, self._height), Image.BILINEAR)
#         frame = np.array(img, dtype=np.uint8)
#         frame = frame[:, :, None]
#         return frame

# class ImageToPyTorchIdentity:
#     def __call__(self, observation):
#         return np.moveaxis(observation, 2, 0)  # (H,W,C) -> (C,H,W)

# class NormalizeFloatsIdentity:
#     def __call__(self, observation):
#         return np.array(observation).astype(np.float32) / 255.0

# class FrameBufferIdentity:
#     def __init__(self, num_steps=4, shape=(1, 84, 84)):
#         self.num_steps = num_steps
#         self.shape = shape
#         self.buffer = np.zeros((num_steps, 84, 84), dtype=np.float32)

#     def reset(self):
#         self.buffer = np.zeros((self.num_steps, 84, 84), dtype=np.float32)

#     def __call__(self, observation):
#         observation = np.squeeze(observation, axis=0)
#         self.buffer[:-1] = self.buffer[1:]
#         self.buffer[-1] = observation
#         return self.buffer

# class ObservationProcessor:
#     def __init__(self):
#         self.downsample = FrameDownsampleIdentity()
#         self.to_tensor = ImageToPyTorchIdentity()
#         self.normalizer = NormalizeFloatsIdentity()
#         self.frame_buffer = FrameBufferIdentity(num_steps=4)

#     def reset(self):
#         self.frame_buffer.reset()

#     def process(self, obs):
#         obs = self.downsample(obs)
#         obs = self.to_tensor(obs)
#         obs = self.normalizer(obs)
#         obs = self.frame_buffer(obs)
#         return obs

# # Agent
# class Agent(object):
#     def __init__(self):
#         self.action_space = gym.spaces.Discrete(12)
#         self.device = torch.device("cpu")  # 強制用 CPU
        
#         self.agent = DQNAgent((4, 84, 84), 12, device=self.device)
        
#         checkpoint = torch.load("./model.pth", map_location=self.device)
#         self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])
        
#         self.obs_processor = ObservationProcessor()

#     def act(self, observation):
#         observation = self.obs_processor.process(observation)
#         action = self.agent.get_action(observation, 0.01)
#         return action

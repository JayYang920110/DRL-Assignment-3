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
class FrameDownsampleIdentity:
    def __init__(self, width=84, height=84):
        self._width = width
        self._height = height

    def __call__(self, observation):
        img = Image.fromarray(observation)
        img = img.convert('L')
        img = img.resize((self._width, self._height), Image.BILINEAR)
        frame = np.array(img, dtype=np.uint8)
        return frame

class ImageToPyTorchIdentity:
    def __call__(self, observation):
        return np.moveaxis(observation, 2, 0)  # (H,W,C) -> (C,H,W)

class NormalizeFloatsIdentity:
    def __call__(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

class FrameBufferIdentity:
    def __init__(self, num_steps=4, shape=(1, 84, 84)):
        self.num_steps = num_steps
        self.shape = shape
        self.buffer = np.zeros((num_steps,) + shape, dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros((self.num_steps,) + self.shape, dtype=np.float32)

    def __call__(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ObservationProcessor:
    def __init__(self):
        self.downsample = FrameDownsampleIdentity()
        self.to_tensor = ImageToPyTorchIdentity()
        self.normalizer = NormalizeFloatsIdentity()
        self.frame_buffer = FrameBufferIdentity(num_steps=4)

    def reset(self):
        self.frame_buffer.reset()

    def process(self, obs):
        obs = self.downsample(obs)
        obs = self.to_tensor(obs)
        obs = self.normalizer(obs)
        obs = self.frame_buffer(obs)
        return obs

# Agent
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cpu")  # 強制用 CPU
        
        self.agent = DQNAgent((4, 84, 84), 12, device=self.device)
        
        checkpoint = torch.load("models_temp/model.pth", map_location=self.device)
        self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])
        
        self.obs_processor = ObservationProcessor()

    def act(self, observation):
        observation = self.obs_processor.process(observation)
        action = self.agent.get_action(observation, 0.2)
        return action

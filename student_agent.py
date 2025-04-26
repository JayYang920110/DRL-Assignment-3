import gym
import torch
from torch import nn
import numpy as np
import random
# Gym is an OpenAI toolkit for RL
import gym

class QNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNet, self).__init__()
        self.input_shape = input_shape
        self.num_actions = n_actions

        # Shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self._feature_size = self._get_feature_size()

        # Dueling: separate fully connected streams
        self.fc_value = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # V(s)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)  # A(s, a)
        )

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        value = self.fc_value(x)               # shape: (batch, 1)
        advantage = self.fc_advantage(x)       # shape: (batch, n_actions)

        # Combine V(s) and A(s,a) into Q(s,a)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def _get_feature_size(self):
        with torch.no_grad():
            x = self.features(torch.zeros(1, *self.input_shape))
            return x.view(1, -1).size(1)
        

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-4, buffer_size=100000, batch_size=128, target_update_freq=1000, init_learning=20000, tau=1e-3):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.    
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network
        self.qnet_local = QNet(state_size, action_size)
        self.qnet_target = QNet(state_size, action_size)
    def get_action(self, state, eps=0.):
        if random.random() > eps:
            # state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device) #(1, 4, 84, 84)
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0)
            self.qnet_local.eval()
            with torch.no_grad():
                action_value = self.qnet_local(state)
            return action_value.max(1)[1].item()
        else:
            return random.choice(np.arange(self.action_size))
        
# class FrameDownsampleIdentity:
#     def __init__(self, width=84, height=84):
#         self._width = width
#         self._height = height

#     def __call__(self, observation):
#         frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
#         return frame[:, :, None]
from PIL import Image
import numpy as np

class FrameDownsampleIdentity:
    def __init__(self, width=84, height=84):
        self._width = width
        self._height = height

    def __call__(self, observation):

        img = Image.fromarray(observation)
        img = img.convert('L')
        img = img.resize((self._width, self._height), Image.BILINEAR)
        frame = np.array(img, dtype=np.uint8)
        return frame[:, :, None]
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
    

agent = DQNAgent((4,84,84), 12)
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.agent = DQNAgent(4, 12)  # 你原本就有
        checkpoint = torch.load("models_normal_99999975/model.pth")
        self.agent.qnet_local.load_state_dict(checkpoint['qnet_local'])
        self.obs_processor = ObservationProcessor()
        
    def act(self, observation):

        
        observation = self.obs_processor.process(observation)
        action = self.agent.get_action(observation, 0.2)
        return action
    

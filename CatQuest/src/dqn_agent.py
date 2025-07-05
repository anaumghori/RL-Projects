import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from configs.config import *

class DQN(nn.Module):
    # Deep Q-Network with fully connected layers
    def __init__(self, state_size: int, action_size: int, hidden_size: int = HIDDEN_SIZE):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayMemory:
    # Experience replay buffer for storing transitions
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    # DQN agent with experience replay and target network
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        
        # Training parameters
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.update_target_freq = TARGET_UPDATE_FREQ
        self.learn_step = 0
        
        # Initialize target network
        self.update_target_network()
    
    # Select action using epsilon-greedy policy
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()
    
    # Store experience in replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    # Train the network using experience replay
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.learn_step += 1
        if self.learn_step % self.update_target_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Update target network weights
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # Save model weights
    def save_model(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, filepath)
    
    # Load model weights
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']
    
    # Get current epsilon value
    def get_epsilon(self) -> float:
        return self.epsilon
    
    # Set epsilon value
    def set_epsilon(self, epsilon: float):
        self.epsilon = max(epsilon, self.epsilon_min)
    
    # Get Q-values for a state
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy().flatten()
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from configs.config import *
from src.environment import CatPuzzleEnv
from src.dqn_agent import DQNAgent

class Trainer:
    def __init__(self):
        self.env = CatPuzzleEnv()
        self.agent = DQNAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_action_space_size(),
            device='cpu'
        )
        
        self.episode_rewards = []
        self.success_count = 0
        
    def train(self, episodes: int = 500):
        for episode in range(episodes):
            state = self.env.reset()
            state_flat = self.env.get_state_flat()
            episode_reward = 0
            
            while not self.env.done:
                # Select and execute action
                action = self.agent.select_action(state_flat, training=True)
                next_state, reward, done, info = self.env.step(action)
                next_state_flat = self.env.get_state_flat()
                
                # Store experience and train
                self.agent.remember(state_flat, action, reward, next_state_flat, done)
                self.agent.replay()
                
                state_flat = next_state_flat
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            
            # Progress update every 100 episodes
            if (episode + 1) % 100 == 0:
                recent_avg = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{episodes} - Avg Reward: {recent_avg:.1f}")
        
        # Final training summary and plot
        self._show_training_summary()
        self._save_plot()
    
    def test_agent(self, episodes: int = 5):
        successes = 0
        
        for i in range(episodes):
            print(f"\n--- Test Episode {i+1}/{episodes} ---")
            
            state = self.env.reset()
            state_flat = self.env.get_state_flat()
            total_reward = 0
            step_count = 0
            
            # Show initial state
            print("Starting position:")
            self.env.render()
            
            while not self.env.done:
                action = self.agent.select_action(state_flat, training=False)
                next_state, reward, done, info = self.env.step(action)
                state_flat = self.env.get_state_flat()
                total_reward += reward
                step_count += 1
                
                # Show move
                print(f"Move {step_count}: {ACTION_NAMES[action]}")
                self.env.render()
                
                if done:
                    break
            
            # Episode summary
            success = self.env.is_goal_reached()
            if success:
                successes += 1
                print(f"SUCCESS! Reward: {total_reward:.1f}, Steps: {step_count}")
            else:
                print(f"Failed. Reward: {total_reward:.1f}, Steps: {step_count}")
        
        # Final test summary
        success_rate = (successes / episodes) * 100
        print(f"\nTest Results: {successes}/{episodes} successful ({success_rate:.0f}%)")
    
    def _show_training_summary(self):
        print(f"\n Training Summary:")
        print(f"   Total Episodes: {len(self.episode_rewards)}")
        print(f"   Average Reward: {np.mean(self.episode_rewards):.1f}")
        print(f"   Best Reward: {np.max(self.episode_rewards):.1f}")
        print(f"   Final Epsilon: {self.agent.get_epsilon():.2f}")
    
    def _save_plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
        
        # Add moving average if enough episodes
        if len(self.episode_rewards) >= 50:
            moving_avg = []
            for i in range(49, len(self.episode_rewards)):
                moving_avg.append(np.mean(self.episode_rewards[i-49:i+1]))
            plt.plot(range(49, len(self.episode_rewards)), moving_avg, 
                    color='red', linewidth=2, label='Moving Average (50 episodes)')
            plt.legend()
        
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('results/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Training plot saved to 'results/training_progress.png'")
    
    def save_model(self, filepath: str):
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        self.agent.load_model(filepath)
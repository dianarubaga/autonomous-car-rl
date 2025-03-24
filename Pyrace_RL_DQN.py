import gymnasium as gym
import gym_race
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import glob
import pygame  # for event handling

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -------------------------------
# Define the DQN Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Two hidden layers with 64 units each (adjust as needed)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# Define a basic Replay Memory
# -------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -------------------------------
# Define the DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Create the policy network
        self.policy_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Experience replay memory
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.policy_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -------------------------------
# Main Loop: Training / Testing / Best Mode
# -------------------------------
def main():
    # Modes and parameters:
    TRAINING_MODE = True
    BEST_MODE = True             # If True, train until TARGET_REWARD is achieved.
    TARGET_REWARD = 10000
    #LOAD_LATEST = True

    # Create the environment.
    env = gym.make("Pyrace-v1").unwrapped

    # Directory for saving models and memory.
    save_dir = "models_DQN"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters.
    num_episodes   = 3500
    max_t          = 2000
    gamma          = 0.99
    lr             = 1e-3
    epsilon        = 1.0
    epsilon_min    = 0.01
    epsilon_decay  = 0.995
    memory_capacity= 10000
    batch_size     = 64

    agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity)
    start_episode = 0

    # Check for existing best files.
    best_files = glob.glob(os.path.join(save_dir, "dqn_*_best.pth"))
    if BEST_MODE and best_files:
        best_episodes = []
        for file in best_files:
            basename = os.path.basename(file)
            try:
                # Expected filename format: dqn_{episode}_best.pth
                ep = int(basename.split('_')[1])
                best_episodes.append(ep)
            except Exception as e:
                print(f"Could not parse episode from {basename}: {e}")
        if best_episodes:
            best_episode = max(best_episodes)
            best_model_path = os.path.join(save_dir, f"dqn_{best_episode}_best.pth")
            best_memory_path = os.path.join(save_dir, f"memory_{best_episode}_best.npy")
            agent.policy_net.load_state_dict(torch.load(best_model_path))
            loaded_memory = np.load(best_memory_path, allow_pickle=True)
            agent.memory.memory = deque(loaded_memory.tolist(), maxlen=memory_capacity)
            print(f"Best model and memory found. Loaded {best_model_path} and {best_memory_path}.")
            print("Running the car game with the best model in test mode.")
            TRAINING_MODE = False  # Switch to test mode.
            start_episode = best_episode

    # For non-training (test) mode, force the agent to act greedily.
    if not TRAINING_MODE:
        agent.epsilon = 0.0
        print("Running in test (non-training) mode. The game will be displayed continuously.")

    rewards_list = []
    terminate = False  # Flag for user exit.

    # Main loop.
    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0

        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            if TRAINING_MODE:
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step(batch_size)

            state = next_state
            total_reward += reward

            # Always render the game in test mode; in training mode, render every 100 episodes.
            if not TRAINING_MODE or episode % 100 == 0:
                env.set_msgs([
                    f'Episode: {episode}',
                    f'Time step: {t}',
                    f'Reward: {total_reward:.0f}'
                ])
                env.render()

                # Check for exit events.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        terminate = True
                        break
                if terminate:
                    break

            if done or terminate:
                break

        rewards_list.append(total_reward)

        if TRAINING_MODE:
            agent.update_epsilon()
            print(f"Episode {episode} Total Reward: {total_reward} Epsilon: {agent.epsilon:.3f}")

            # Save periodically if not in BEST_MODE.
            if not BEST_MODE and episode % 200 == 0 and episode != 0:
                periodic_model_path = os.path.join(save_dir, f"dqn_{episode}.pth")
                torch.save(agent.policy_net.state_dict(), periodic_model_path)
                periodic_memory_path = os.path.join(save_dir, f"memory_{episode}.npy")
                np.save(periodic_memory_path, np.array(agent.memory.memory, dtype=object))
                print(f"Saved model and memory at episode {episode}")

            # In BEST_MODE, check if the target reward is achieved.
            if BEST_MODE and total_reward >= TARGET_REWARD:
                best_model_path = os.path.join(save_dir, f"dqn_{episode}_best.pth")
                best_memory_path = os.path.join(save_dir, f"memory_{episode}_best.npy")
                torch.save(agent.policy_net.state_dict(), best_model_path)
                np.save(best_memory_path, np.array(agent.memory.memory, dtype=object))
                print(f"Target reward achieved at episode {episode} with reward {total_reward}.")
                print(f"Saved best model to {best_model_path} and memory to {best_memory_path}.")
                break  # Stop training.
        else:
            print(f"Test Episode {episode} Total Reward: {total_reward}")

        if terminate:
            print("Termination requested. Exiting...")
            break

    # In training mode (if not BEST_MODE), save the final model.
    if TRAINING_MODE and not BEST_MODE:
        final_model_path = os.path.join(save_dir, "dqn_model_final.pth")
        torch.save(agent.policy_net.state_dict(), final_model_path)
        final_memory_path = os.path.join(save_dir, "final_memory.npy")
        np.save(final_memory_path, np.array(agent.memory.memory, dtype=object))
    env.close()

if __name__ == "__main__":
    main()

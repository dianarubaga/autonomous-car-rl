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
        # Two hidden layers with 64 units each
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
    def __init__(self):
        self.memory = []

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
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay):
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
        self.memory = ReplayMemory()

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
def check_best_model_exists(save_dir):
    best_files = glob.glob(os.path.join(save_dir, "dqn_*_best.pth"))
    return len(best_files) > 0

def test_perfect_model(env, agent, target_reward):
    """Test a model with rendering to ensure it can complete the circuit perfectly."""
    print("Testing model with rendering...")
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0
    perfect = True

    for t in range(2000):  # Use the same max_t as training
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        # Always render during testing
        env.set_msgs([
            f'Testing perfect model',
            f'Time step: {t}',
            f'Reward: {total_reward:.0f}'
        ])
        env.render()

        state = next_state
        total_reward += reward

        # Check for exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False, total_reward

        if done:
            break

    # Check if the model completed the circuit with perfect reward
    perfect = total_reward >= target_reward
    return perfect, total_reward

def main(best_mode=False, target_reward=10000):
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
    batch_size     = 64

    agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay)
    start_episode = 0

    # Check for existing best files.
    if check_best_model_exists(save_dir):
        best_mode = True
        best_files = glob.glob(os.path.join(save_dir, "dqn_*_best.pth"))
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
            agent.memory.memory = loaded_memory.tolist()
            print(f"Best model and memory found. Loaded {best_model_path} and {best_memory_path}.")
            print("Running the car game with the best model in test mode.")
            start_episode = best_episode

    # If in best mode, force the agent to act greedily
    if best_mode:
        agent.epsilon = 0.0
        print("Running in best mode. The game will be displayed continuously.")

    rewards_list = []
    terminate = False  # Flag for user exit.

    # Main loop.
    episode = start_episode
    while episode < num_episodes:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0

        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            if not best_mode:  # Only train and store transitions if not in best mode
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step(batch_size)

            state = next_state
            total_reward += reward

            # Always render the game in best mode; in training mode, render every 100 episodes
            if best_mode or episode % 100 == 0:
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

        if not best_mode:  # Only update epsilon and save models if not in best mode
            agent.update_epsilon()
            print(f"Episode {episode} Total Reward: {total_reward} Epsilon: {agent.epsilon:.3f}")
            episode += 1  # Only increment episode counter in training mode

            # Save periodically
            if episode % 200 == 0 and episode != 0:
                periodic_model_path = os.path.join(save_dir, f"dqn_{episode}.pth")
                torch.save(agent.policy_net.state_dict(), periodic_model_path)
                periodic_memory_path = os.path.join(save_dir, f"memory_{episode}.npy")
                np.save(periodic_memory_path, np.array(agent.memory.memory, dtype=object))
                print(f"Saved model and memory at episode {episode}")

            # Check if target reward is achieved
            if total_reward >= target_reward:
                print(f"Potential perfect model found with reward {total_reward}. Testing with rendering...")

                # Test the model with rendering
                perfect, test_reward = test_perfect_model(env, agent, target_reward)

                if perfect:
                    best_model_path = os.path.join(save_dir, f"dqn_{episode}_best.pth")
                    best_memory_path = os.path.join(save_dir, f"memory_{episode}_best.npy")
                    torch.save(agent.policy_net.state_dict(), best_model_path)
                    np.save(best_memory_path, np.array(agent.memory.memory, dtype=object))
                    print(f"Perfect model confirmed! Saved to {best_model_path} and memory to {best_memory_path}.")
                    # After saving the best model, run main again with best_mode=True
                    env.close()
                    main(best_mode=True, target_reward=target_reward)
                    return
                else:
                    print(f"Model failed perfect test with reward {test_reward}. Continuing training...")
                    # Continue training without saving the model
        else:
            print(f"Best Mode Episode {episode} Total Reward: {total_reward}")

        if terminate:
            print("Termination requested. Exiting...")
            break

    # Save the final model if in training mode
    if not best_mode:
        final_model_path = os.path.join(save_dir, "dqn_model_final.pth")
        torch.save(agent.policy_net.state_dict(), final_model_path)
        final_memory_path = os.path.join(save_dir, "final_memory.npy")
        np.save(final_memory_path, np.array(agent.memory.memory, dtype=object))
    env.close()

if __name__ == "__main__":
    main()

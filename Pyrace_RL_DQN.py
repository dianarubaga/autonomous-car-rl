import os, math, random, time, json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_race
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

# -------------------- CONFIGURATION --------------------
EXPERIMENT_NAME = 'DQN_CarAgent_v01'
TOTAL_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 2000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_CAPACITY = 100_000
MIN_MEMORY_SIZE = 1000

COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pygame.init()

# -------------------- NEURAL NETWORK --------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, inputs):
        return self.layers(inputs)

# -------------------- EXPERIENCE REPLAY --------------------
class ExperienceMemory:
    def __init__(self, max_capacity):
        self.memory = deque(maxlen=max_capacity)

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(COMPUTE_DEVICE),
            torch.tensor(actions).to(COMPUTE_DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(COMPUTE_DEVICE),
            torch.tensor(next_states, dtype=torch.float32).to(COMPUTE_DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(COMPUTE_DEVICE)
        )

    def __len__(self):
        return len(self.memory)

# -------------------- CAR AGENT --------------------
class CarAgent:
    def __init__(self, input_dim, action_dim):
        self.model = NeuralNetwork(input_dim, action_dim).to(COMPUTE_DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = ExperienceMemory(MEMORY_CAPACITY)
        self.batch_size = BATCH_SIZE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = 1.0
        self.top_reward = -np.inf
        self.current_episode = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.model.layers[-1].out_features - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(COMPUTE_DEVICE)
        return torch.argmax(self.model(state_tensor)).item()

    def optimize_model(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_batch(self.batch_size)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, filepath=f"models_{EXPERIMENT_NAME}/best_checkpoint.pth"):
        torch.save(self.model.state_dict(), filepath)
        metadata = {
            "episode": self.current_episode,
            "epsilon": self.epsilon,
            "top_reward": self.top_reward
        }
        with open(filepath.replace(".pth", ".json"), "w") as file:
            json.dump(metadata, file)

    def load_checkpoint(self, filepath=f"models_{EXPERIMENT_NAME}/best_checkpoint.pth"):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=COMPUTE_DEVICE))
            self.model.eval()
            print("Checkpoint loaded successfully.")

            meta_path = filepath.replace(".pth", ".json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as file:
                    metadata = json.load(file)
                    self.current_episode = metadata["episode"]
                    self.epsilon = metadata["epsilon"]
                    self.top_reward = metadata["top_reward"]
                print(f"Training resumed: Episode {self.current_episode}, Epsilon {self.epsilon:.3f}, Top reward {self.top_reward}")

# -------------------- TRAINING LOOP --------------------
def train_agent(environment, agent, total_episodes=TOTAL_EPISODES):
    environment.set_view(True)
    stagnation_counter = 0

    for episode in range(agent.current_episode, total_episodes):
        state, _ = environment.reset()
        done = False
        episode_reward = 0
        timestep = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = environment.step(action)

            reward += info.get("dist", 0) * 3  # revert to original multiplier
            reward -= 500 if info.get("crash", 0) > 0 else 0
            reward += 500 if info.get("check", 0) > 0 else 0


            # NEW: Small penalty per timestep to encourage faster completion
            reward -= 5

            agent.memory.add_experience(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            timestep += 1

            environment.set_msgs([
                f'Episode: {episode}',
                f'Time steps: {timestep}',
                f'Total reward: {episode_reward:.0f}'
            ])
            environment.render()
            pygame.event.pump()
            time.sleep(0.01)

            if timestep % 2 == 0:
                agent.optimize_model()

        print(f"Episode {episode+1}: Total Reward = {episode_reward:.2f}")

        stagnation_counter = stagnation_counter + 1 if episode_reward == agent.top_reward else 0

        if stagnation_counter >= 3:
            agent.epsilon = 0.8
            print("Epsilon reset due to stagnation.")

        if episode_reward > agent.top_reward:
            agent.top_reward = episode_reward
            agent.save_checkpoint()
            print("New top reward! Checkpoint saved.")

        agent.current_episode += 1
        agent.epsilon = max(0.01, agent.epsilon * 0.99)  # faster decay


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    os.makedirs(f"models_{EXPERIMENT_NAME}", exist_ok=True)
    env = gym.make("Pyrace-v1").unwrapped
    agent = CarAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_checkpoint()

    try:
        train_agent(env, agent)
    except KeyboardInterrupt:
        agent.save_checkpoint()
        print("Training interrupted. Checkpoint saved.")

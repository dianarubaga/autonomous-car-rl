import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import gymnasium as gym
import gym_race
import os
import pygame

# -------------------- 1️⃣ BUILD DQN MODEL --------------------
def build_dqn(state_size, action_size):
    model = Sequential([
        Dense(128, input_dim=state_size, activation='relu'),
        Dense(128, activation='relu'),  
        Dense(64, activation='relu'),  
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# -------------------- 2️⃣ REPLAY BUFFER --------------------
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)

# -------------------- 3️⃣ DQN AGENT --------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = build_dqn(state_size, action_size)
        self.memory = ReplayBuffer()
        self.gamma = 0.99  
        self.epsilon = 0.8  
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997  
        self.batch_size = 128  
        self.best_reward = -float('inf')  
        self.episode = self.load_model()
        self.last_action = None  

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)  
        else:
            q_values = self.model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_values[0])  

        if self.last_action == action:
            action = np.random.choice(self.action_size)  

        self.last_action = action
        return action  

    def train(self):
        if self.memory.size() < self.batch_size:
            return  

        batch_size = min(len(self.memory.memory), self.batch_size * 2)  
        batch = self.memory.sample(batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state[np.newaxis, :], verbose=0)[0]
            if done:
                target[action] = reward
            else:
                next_q_values = self.model.predict(next_state[np.newaxis, :], verbose=0)
                target[action] = reward + self.gamma * np.max(next_q_values[0])

            states.append(state)
            targets.append(target)

        self.model.train_on_batch(np.array(states), np.array(targets))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        episode_str = str(self.episode).zfill(3)  
        model_filename = f"dqn_model_{episode_str}.keras"
        self.model.save(model_filename)
        print(f"✅ Model saved: {model_filename}")

        models = sorted([f for f in os.listdir() if f.startswith("dqn_model_") and f.endswith(".keras")])
        if len(models) > 5:  
            os.remove(models[0])  
            print(f"🗑️ Deleted old model: {models[0]}")

        with open("last_episode.txt", "w") as f:
            f.write(str(self.episode))
        with open("best_reward.txt", "w") as f:
            f.write(str(self.best_reward))

    def load_model(self):
        episode = 0  
        models = sorted([f for f in os.listdir() if f.startswith("dqn_model_") and f.endswith(".keras")])

        if models:
            best_model = models[-1]  
            try:
                self.model.load_weights(best_model)
                print(f"✅ Model loaded: {best_model}")
            except ValueError:
                print("🚨 Model structure has changed! Resetting.")
                os.remove(best_model)

        if os.path.exists("last_episode.txt"):
            with open("last_episode.txt", "r") as f:
                episode = int(f.read().strip())

        if os.path.exists("best_reward.txt"):
            with open("best_reward.txt", "r") as f:
                self.best_reward = float(f.read().strip())

        print(f"✅ Resuming from Episode: {episode}, Best Reward: {self.best_reward}")
        return episode  

# -------------------- 4️⃣ TRAINING FUNCTION --------------------
def simulate(env, agent, episodes=1000):
    env.set_view(True)  
    max_reward = -10_000  
    stuck_count = 0  

    for episode in range(agent.episode, episodes):  
        state, _ = env.reset()
        done = False
        total_reward = 0
        t = 0
        turn_penalty = 0  

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            if "dist" in info:
                reward += info["dist"] * 3  

            if "crash" in info and info["crash"] > 0:
                reward -= 500  

            if "check" in info and info["check"] > 0:
                reward += 500  

         

            agent.memory.store(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            t += 1

            env.set_msgs([
                'SIMULATE',
                f'Episode: {episode}',
                f'Time steps: {t}',
                f'Reward: {total_reward:.0f}'
            ])
            env.render()
            pygame.event.pump()

            if t % 2 == 0:  
                agent.train()
 

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

        if total_reward == agent.best_reward:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= 3:  
            print("🚨 Model stuck! Resetting epsilon.")
            agent.epsilon = 0.8  

        if total_reward > agent.best_reward:
            agent.best_reward = total_reward
            print("🏆 Best model updated!")
            agent.save_model()

# -------------------- 5️⃣ MAIN FUNCTION --------------------
if __name__ == "__main__":
    env = gym.make("Pyrace-v1").unwrapped  
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    simulate(env, agent, episodes=1000)

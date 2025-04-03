# Race Car Machine Learning Exercise

## Deep Q-Network Model Breakdown

1. **Neural Network Architecture (DQN Class)**
```python
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
```
The DQN uses a fully connected neural network with:
- Input layer: Takes the state representation
- Two hidden layers with 64 units each and ReLU activation
- Output layer: Produces Q-values for each possible action

2. **Experience Replay (ReplayMemory Class)**
```python
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
```
The implementation uses experience replay to:
- Store transitions (state, action, reward, next_state, done) in a fixed-size memory
- Sample random batches for training to break temporal correlations
- Improve learning stability by reusing past experiences

3. **DQN Agent Implementation**
The agent combines several key DQN mechanisms:

a) **ε-greedy Action Selection**:
```python
def select_action(self, state):
    if random.random() < self.epsilon:
        return random.randrange(self.action_dim)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item())
```
- Balances exploration (random actions) and exploitation (best actions)
- Epsilon decays over time to reduce exploration gradually

b) **Q-Learning Update**:
```python
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
```
The training process:
- Samples a batch of transitions from replay memory
- Computes current Q-values for the taken actions
- Computes target Q-values using the Bellman equation: R + γ * max(Q(s', a'))
- Updates the network using MSE loss between current and target Q-values

4. **Hyperparameters and Training Configuration**
```python
num_episodes   = 3500
max_t          = 2000
gamma          = 0.99
lr             = 1e-3
epsilon        = 1.0
epsilon_min    = 0.01
epsilon_decay  = 0.995
memory_capacity= 10000
batch_size     = 64
```
The implementation uses standard DQN hyperparameters:
- Discount factor (gamma) of 0.99
- Learning rate of 0.001
- Epsilon decay from 1.0 to 0.01
- Replay memory capacity of 10,000 transitions
- Training batch size of 64

5. **Model Persistence**
The implementation includes functionality to:
- Save the model periodically during training
- Save the best performing model when reaching a target reward
- Load previously trained models for testing or continued training

This implementation follows the core DQN principles from the original DeepMind paper, including experience replay and fixed Q-targets, while adding practical features like model saving/loading and visualization capabilities.

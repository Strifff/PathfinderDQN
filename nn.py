import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNet(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        gridsize,
        gamma,
        entropy,
        entropy_min,
        lr,
    ):
        super(QNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.gridsize = gridsize
        self.gamma = gamma
        self.entropy = entropy
        self.entropy_min = entropy_min
        self.lr = lr

        self.qvalues = np.zeros((gridsize, gridsize))

        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=0.0001
        )
        self.loss_fn = nn.MSELoss()

        self.mem = None

    def forward(self, x):
        if np.random.rand() < self.entropy:
            self.entropy = max(self.entropy_min, self.entropy - 0.00001)
            return np.random.randint(self.output_size)

        x = torch.tensor(x, dtype=torch.float32)
        x = self.net(x)
        x = torch.argmax(x).item()
        return x

    def train(self, batch_size):

        batch = self.mem.sample(batch_size)

        old_states = torch.tensor(
            [transition[0] for transition in batch], dtype=torch.float32
        )
        actions = torch.tensor(
            [transition[1] for transition in batch], dtype=torch.long
        )
        rewards = torch.tensor(
            [transition[2] for transition in batch], dtype=torch.float32
        )
        new_states = torch.tensor(
            [transition[3] for transition in batch], dtype=torch.float32
        )
        dones = torch.tensor([transition[4] for transition in batch], dtype=torch.bool)

        current_q_values = self.net(old_states)

        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.net(new_states)
            max_next_q_values = torch.max(next_q_values, dim=1).values
            target_q_values = rewards + self.gamma * max_next_q_values
            target_q_values[dones] = rewards[dones]

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_q_values(self, new_state, reward, done):
        new_state_tensor = torch.tensor(new_state, dtype=torch.float32)

        with torch.no_grad():
            if done:
                target_q_value = torch.tensor(reward, dtype=torch.float32)
            else:
                next_q_values = self.net(new_state_tensor)
                max_next_q_value = torch.max(next_q_values)
                target_q_value = reward + self.gamma * max_next_q_value
        self.qvalues[-int(new_state[1]) + 7, -int(new_state[0]) + 4] = target_q_value

    def q_value_swepp(self):
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                reward = -0.05
                done = False
                if i == 4 and j == 7:
                    reward = 1
                    done = True
                elif i == 7 and j in range(3, 12):
                    reward = -1
                    done = True

                dx = 4 - i
                dy = 7 - j
                self.set_q_values([dx, dy], reward, done)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

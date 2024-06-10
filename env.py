import numpy as np

from collections import deque

# from mem import ReplayMemory


class GridEnv:
    def __init__(self, size, goal, danger=[]):
        self.size = size
        self.goal = goal
        self.piece = [0, 0]
        self.danger = danger
        self.done_reward = 1
        self.danger_reward = -1
        self.standrard_reward = -0.05
        self.dones = 0
        self.dangers = 0
        self.wr_deque = deque(maxlen=10000)

        self.actions = ["up", "down", "left", "right"]
        self.action_space = len(self.actions)
        self.done = False
        self.it = 0
        self.max_it = 100

        self.nn = None
        self.ani = False
        self.cp = None
        self.episode_reward = 0

        self.mem = None
        self.batch_size = 1024

        self.sweep_array_ma1000 = []
        self.ma1000_deque = deque(maxlen=1000)

    def reset(self):

        def uniform_random_start():
            self.piece = [
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            ]

        def semi_random_corner_start():
            self.piece = [
                np.random.choice([0, 1, self.size - 2, self.size - 1]),
                np.random.choice([0, 1, self.size - 2, self.size - 1]),
            ]

        okay = False
        while not okay:
            uniform_random_start()

            okay = True
            for d in self.danger:
                if self.piece == d:
                    okay = False
                    break
            if self.piece == self.goal:
                okay = False

        self.done = False

        self.it = 0

        if not self.ani:
            if self.cp:
                self.cp.update(self.episode_reward)
            self.ma1000_deque.append(self.episode_reward)
            self.sweep_array_ma1000.append(
                sum(self.ma1000_deque) / len(self.ma1000_deque)
            )

            self.episode_reward = 0
            self.nn.train(self.batch_size)

        if self.ani:
            print(self.stats())

    def stats(self):
        if self.dones + self.dangers == 0:
            return "No stats yet"
        # winrate = self.dones / (self.dones + self.dangers) * 100
        winrate = sum(self.wr_deque) / len(self.wr_deque) * 100
        return f"{winrate:.2f}% winrate, {self.dones} wins, {self.dangers} losses, {self.nn.entropy:.2f} entropy"

    def take_action(self, action):
        if action == "up":
            if self.piece[1] < self.size - 1:
                self.piece[1] += 1
        elif action == "down":
            if self.piece[1] >= 1:
                self.piece[1] -= 1
        elif action == "left":
            if self.piece[0] >= 1:
                self.piece[0] -= 1
        elif action == "right":
            if self.piece[0] < self.size - 1:
                self.piece[0] += 1
        else:
            raise ValueError("Invalid action")

        assert self.piece[0] >= 0 and self.piece[0] < self.size
        assert self.piece[1] >= 0 and self.piece[1] < self.size

    def reward(self):
        if self.piece == self.goal:
            self.done = True
            self.dones += 1
            self.wr_deque.append(1)
            return self.done_reward
        elif self.piece in self.danger:
            self.done = True
            self.dangers += 1
            self.wr_deque.append(0)
            if self.ani:
                print("Danger!")
            return self.danger_reward
        elif self.it >= self.max_it:
            self.done = True
            self.dangers += 1
            self.wr_deque.append(0)
            if self.ani:
                print("Timeout!")
            return self.danger_reward * 10
        else:
            return self.standrard_reward

    def state(self):
        return self.piece

    def step(self):

        if self.done:
            self.reset()

        dx = self.goal[0] - self.piece[0]
        dy = self.goal[1] - self.piece[1]
        old_state = [dx, dy]

        action_index = self.nn(old_state)

        assert action_index >= 0 and action_index < self.action_space
        assert self.actions[action_index] in self.actions
        action = self.actions[action_index]

        self.take_action(action)

        reward = self.reward()
        self.episode_reward += reward

        # new_state = self.state().copy()

        dx = self.goal[0] - self.piece[0]
        dy = self.goal[1] - self.piece[1]
        new_state = [dx, dy]

        obj = {
            "old_state": old_state,
            "action": action,
            "reward": reward,
            "new_state": new_state,
            "done": self.done,
        }

        # self.nn.train(old_state, action_index, reward, new_state, self.done)

        self.nn.set_q_values(new_state, reward, self.done)

        self.mem.push((old_state, action_index, reward, new_state, self.done))

        self.it += 1

        return obj

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from collections import deque


class GridAnimator:
    def __init__(self, size, qnet, env, train=False):
        self.size = size
        self.qnet = qnet
        self.env = env
        self.train = train
        self.it = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.image = self.ax.imshow(self.qnet.qvalues, cmap="coolwarm", vmin=-1, vmax=1)
        self.ax.set_title("Q-Values Heatmap")

        cbar_ax = self.fig.add_axes([0.15, 0.05, 0.7, 0.03])
        cbar = self.fig.colorbar(self.image, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Q-Value")

        self.texts = []
        for i in range(size):
            row = []
            for j in range(size):
                text = self.ax.text(
                    j,
                    i,
                    f"{self.qnet.qvalues[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
                row.append(text)
            self.texts.append(row)

        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    def update(self, frame):
        obj = self.env.step()

        loc = self.env.piece

        q_values = self.qnet.qvalues.copy()

        norm_q_values = (q_values + 1) / 2

        cmap = plt.get_cmap("coolwarm")
        color_array = cmap(norm_q_values)

        color_array[loc[1], loc[0]] = [1, 1, 0, 1]

        self.image.set_array(color_array)

        for i in range(self.size):
            for j in range(self.size):
                self.texts[i][j].set_text(f"{self.qnet.qvalues[i, j]:.2f}")
        return [self.image] + [text for row in self.texts for text in row]

    def on_draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def animate(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=1,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()


class ContinuousPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.maxlen = 10000
        self.rewards = deque(maxlen=self.maxlen)
        self.ma500 = deque(maxlen=self.maxlen)

    def update(self, reward):
        # print("Updating plot with reward:", reward)  # Debug print
        self.rewards.append(reward)
        ma_len = int(min(len(self.rewards), 500))

        self.ma500.append(sum(list(self.rewards)[-ma_len:]) / ma_len)
        self.ax.clear()
        self.ax.scatter(range(len(self.rewards)), self.rewards, s=5, label="Reward")
        # self.ax.plot(self.rewards, label="Reward")

        self.ax.plot(
            self.ma500[-1] * np.ones(len(self.rewards)),
            label="MA500 last",
            linewidth=2,
            color="black",
        )
        self.ax.plot(self.ma500, label="MA500", linewidth=1.5, color="red")
        self.ax.legend()
        self.fig.canvas.draw()
        plt.pause(0.0000001)  # Ensure plot is updated in interactive environments
        return []

    def show(self):
        plt.show()

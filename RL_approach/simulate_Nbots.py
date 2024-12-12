

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation
import random 

class uBotsSim(object):
    def __init__(self, N, XMIN = -10, XMAX = 10, YMIN = -10, YMAX = 10, dt = 0.1):
        self.N = N
        self.XMIN = XMIN
        self.XMAX = XMAX
        self.YMIN = YMIN
        self.YMAX = YMAX
        self.dt = dt
        self.positions = np.random.uniform([XMIN, YMIN], [XMAX, YMAX], (N, 2))
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.XMIN, self.XMAX)
        self.ax.set_ylim(self.YMIN, self.YMAX)
        self.scat = self.ax.scatter(self.positions[:, 0], self.positions[:, 1])

    def animate(self, frame):
        self.step(self.f, self.alpha)
        self.scat.set_offsets(self.positions)
        return self.scat,

    def run_animation(self, f, alpha, frames=200, interval=50):
        self.f = f
        self.alpha = alpha
        print(f'f,alpha={self.f, self.alpha}')
        ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, blit=True)
        plt.show()



    def v_i(self, f):
        lookup_table = [
            [2.72, 4.06, 5.80, 6.81, 9.07, 9.46, 11.32, 11.95, 14.11, 14.49, 16.15, 16.49, 17.30, 17.09, 18.35, 19.68, 19.45, 21.39, 22.50, 23.65],
            [16.62, 27.30, 37.71, 48.13, 58.72, 66.67, 78.15, 84.48, 96.43, 108.05, 119.22, 120.53, 127.00, 133.90, 131.50, 151.17, 153.06, 161.49, 170.00, 170.95]
        ]
        if self.N > 2:
            print("Warning: Number of bots is greater than 2. Replicating the lookup table for the first 2 bots.")
            lookup_table = lookup_table * (self.N // 2 + 1)
        return np.array([np.interp(f, range(1, 21), lookup_table[i]) for i in range(self.N)])


    def step(self, f, alpha):
        self.f = f
        self.alpha = alpha
        new_positions = []
        speeds = self.v_i(f)
        for i, pos in enumerate(self.positions):
            dx = speeds[i] * self.dt * np.cos(alpha)
            dy = speeds[i] * self.dt * np.sin(alpha)
            new_pos = pos + np.array([dx, dy])
            new_pos[0] = np.clip(new_pos[0], self.XMIN, self.XMAX)
            new_pos[1] = np.clip(new_pos[1], self.YMIN, self.YMAX)
            new_positions.append(new_pos)
        self.positions = np.array(new_positions)

if __name__ == "__main__":
    N = 2
    sim = uBotsSim(N)
    sim.run_animation(0.5, np.pi / 4)



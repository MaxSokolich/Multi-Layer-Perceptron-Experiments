from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(data):
    plt.plot(data[:,0], data[:,2])
    plt.show()

def main():
    dataset_path = Path("dataset")
    states = np.load(dataset_path / "all_pos.npy")
    fs = np.load(dataset_path / "all_frq.npy")
    alphas = np.load(dataset_path / "all_alpha.npy")

    actions = np.column_stack((fs, alphas))
    plot_trajectories(states)

if __name__ == '__main__':
    main()
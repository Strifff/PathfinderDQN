import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def plot_by_parameter():
    folder = "sweeps/uniform_layers"

    all_files = os.listdir(folder)
    # only pickle files
    all_files = [file for file in all_files if file.endswith(".pkl")]

    # keep files with different value after parameter

    # f"sweeps/GAMMA_{param}_ENTROPY_MIN_{ENTROPY_MIN}_LR_{LR}_BATCH_{BATCH_SIZE}_mem_{mem_capacity}.pkl"

    fig, ax = plt.subplots()  # Create a figure and an axes
    # make plot big
    fig.set_size_inches(18.5 / 2, 10.5 / 2, forward=True)

    for file in all_files:
        with open(f"{folder}/{file}", "rb") as f:
            data = pickle.load(f)
            ax.plot(data, label=file)  # Plot data with a label for the legend

    ax.legend()  # Add a legend to the plot
    # legend at lower right
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.grid()
    plt.show()


plot_by_parameter()

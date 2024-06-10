import numpy as np
import tqdm as tqdm
import os, json

from collections import deque

from nn import QNet
from anim import GridAnimator, ContinuousPlotter
from env import GridEnv
from mem import ReplayMemory

import pickle


def main():
    SIZE = 15
    GOAL = [4, 7]
    DANGER = [[7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11]]

    GAMMA = 0.99
    ENTROPY = 0.3  # no sweep needed
    ENTROPY_MIN = 0.1
    LR = 0.0001
    BATCH_SIZE = 1024
    MEM_CAP = 10**6

    #! GAMMA
    # sweep_parameter = [0.8, 0.9, 0.95, 0.99] #! 0.99 best
    #! LR
    # sweep_parameter = [1e-6, 1e-5, 1e-4, 1e-3] #! 1e-4 best
    #! ENTROPY_MIN
    # sweep_parameter = [0, 0.01, 0.05, 0.1, 0.25] #! 0.1 best
    #! BATCH_SIZE
    # sweep_parameter = [256, 512, 1024, 2048, 4096] #! hard to say, more is better but slower
    #! MEM_CAP
    # sweep_parameter = [10**3, 10**4, 10**5, 10**6] #! 10**6 best

    #! Hidden sizes
    # sweep_parameter = [[64, 32], [128, 64], [256, 128], [512, 256], [1024, 512]]
    # sweep_parameter = [
    #     [128, 64],
    #     [128, 64, 32],
    #     [256, 128, 64], # best
    #     [512, 256, 128],
    #     [1024, 512, 256],
    #     [2048, 1024, 512],
    # ]

    # sweep_parameter = [
    #     [256, 128, 64],
    #     [32, 32, 32],
    #     [64, 64, 64],
    #     [128, 128, 128],
    #     [256, 256, 256],
    # ]
    HS = [[128, 128, 128]]
    sweep_parameter = HS

    for param in sweep_parameter:
        # GAMMA = param
        # LR = param
        # ENTROPY_MIN = param
        # BATCH_SIZE = param
        # MEM_CAP = param
        # sweep_path = f"sweeps/GAMMA_{GAMMA}_ENTROPY_MIN_{ENTROPY_MIN}_LR_{LR}_BATCH_{BATCH_SIZE}_MEM_{MEM_CAP}.pkl"

        HS = param
        sweep_path = f"sweeps/HS_{HS}_weight_decay_00001.pkl"

        env = GridEnv(SIZE, GOAL, DANGER)
        mem = ReplayMemory(MEM_CAP)

        # qnet = QNet(2, 4, [256, 128, 64, 32], SIZE, GAMMA, ENTROPY, ENTROPY_MIN, LR)
        qnet = QNet(2, 4, HS, SIZE, GAMMA, ENTROPY, ENTROPY_MIN, LR)
        qnet.mem = mem
        env.mem = mem
        env.nn = qnet
        env.batch_size = BATCH_SIZE

        # model_name = "100wr_1Mit_09g_001ent_000005lr.pt"
        # model_name = "100wr_1Mit_09g_010ent_000010lr.pt"
        # model_name = "100wr_1Mit_09g_025ent_000010lr.pt"
        # model_name = "100wr_1Mit_099g_025ent_000010lr.pt"

        model_name = "tester.pt"

        model_name = "models/" + model_name

        ani = True

        if ani:
            if os.path.exists(model_name):
                qnet.load(model_name)

            qnet.entropy = 0
            qnet.q_value_swepp()
            env.ani = True

            animator = GridAnimator(SIZE, qnet, env)
            animator.animate()

        else:

            # cp = ContinuousPlotter()
            # env.cp = cp
            if model_name == "models/tester.pt":
                os.remove(model_name)

            if os.path.exists(model_name):
                qnet.load(model_name)

            for a in tqdm.tqdm(range(int(500)), ncols=100, desc="Training"):

                stats = env.stats()
                tqdm.tqdm.write(stats)
                for _ in range(1000):
                    env.step()

                qnet.save(model_name)

            # take sweep values
            values = env.sweep_array_ma1000
            # save with pickle
            with open(sweep_path, "wb") as f:
                pickle.dump(values, f)


if __name__ == "__main__":
    main()

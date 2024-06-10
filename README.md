# Pathfinder Reinforcement Learning using Q-values
A simple discrete pathfinder using Q-learning for environemnt with a goal and a dangerous wall.

The algorithm learns how the actor should traverse the environment by prediction the values of each of the 4 discrete actions, up, down, left, and right.

When the actor reaches the goal it gets 1 point and if it moves into the dangerours wall it gets -1 point.

## Showcase of the trained actor
![Gif of finished pathfinder](images/GIF.gif)

## Optimizations

### Gamma

![Gamma optimization](images/opt_gamma.png)

### Learning Rate

![Learning rate optimization](images/opt_lr.png)

### Minimum entropy

![Minimum entropy optimization](images/opt_entropy.png)

### Batch size

![Batch size optmization](images/opt_batchsize.png)

### Replay memory size

![Replay memory optimization](images/opt_memcap.png)

### L2 regularzation 

![L2 regularzation](images/opt_weight_decay.png)

### Neural network hidden sizes

#### 2 hidden layers

![2 hidden layers](images/opt_2deep.png)

#### 3 hidden layers

![3 hidden layers](images/opt_3deep.png)

#### Uniform hidden layers

![Uniform hidden layers](images/opt_uniform.png)


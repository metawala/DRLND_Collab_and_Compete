[//]: # (Image References)

[image1]: ./ReportImages/ActorModel.png "Actor Model"
[image2]: ./ReportImages/ResultGraph.png "ResultGraph"
[image3]: ./ReportImages/CriticModel.png "Critic Model"
[image4]: ./ReportImages/HyperParameters.png "Hyper Parameters"
[image5]: ./ReportImages/EnvCharac.png "Environment Characteristics"

# Project Report : Collaboration and Competition
This project report is in relation with the third project in DRLND course - Continuous Control. The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm - 4 action space. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Below are the characteristics of the environment:

![Environment Characteristics][image5]

## Learning Algorithm:
We make use of the Deep Deterministic Policy Gradients - DDPG - algorithm for this project with modification to make it suitable for multiagent environment.

As per DDPG which makes use of an Actor and a Critic NN, we use 2 DNN models. We use a similar network architechture for both of them:

**ACTOR**:
**State --> BatchNorm --> 128 --> ReLU --> 64 --> ReLU --> BatchNorm --> action --> tanh**

Here is screenshot for our Actor Model:

![Actor Model][image1]

this above is for Actor, our critic varies only slightly.

**CRITIC**:
**State --> BatchNorm --> 128 --> Relu --> 64 --> Relu --> action**

While not much different, here is how the Critic model looks like:

![Critic Model][image3]

We use the following hyperparameters:
1. Gamma - Discount Factor = 0.99
2. Learning rate for Actor = 1e-4
3. Learning rate for Critic = 1e-3

In addition, we also `SoftUpdates` with a `TAU = 1e-3` in order to calculate target values for both Actor and Critic.

Comprehensively, all hyperparameters are as such:

![Hyper Parameters][image4]

### Experience Replay:
With a `BUFFERSIZE = int(1e6)` and a `BATCHSIZE = 1024` we create a data container - the replay buffer. We batch from this random indepenent samples to stabally train the network.

## Plots of Rewards:
Initially without a random seed and batch normalization the model took way too much time to train. However, after adding a random seed and updating the initializer and adding batch normalization we can see that the model was solved in **`1027 EPISODES`**.

Below is a graph of the reward as plotted agains episodes:
![Result Graph][image2]

## Ideas for Future Work:
This project implements the RelayBuffer but there is room for trying out other things:
1. Could try playing around with the multi-agent aspect considering this project leaves it rather a bit simple.
2. Trying out MADDPG
3. I hardly played with the hyperparameters in this one, but I would want to experiment with the hyperparameteres to see how that affects learning and if I can improving the network.
4. Try out the Soccer problem as well.
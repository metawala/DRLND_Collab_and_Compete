import copy
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
from AgentModel.model import Actor, Critic

###
# HYPERPARAMETERS
###
BUFFERSIZE     = int(1e6)
BATCHSIZE      = 1024
GAMMA          = 0.99
TAU            = 1e-3
ACTORLEARNING  = 1e-4
CRITICLEARNING = 1e-3
WEIGHTDECAY    = 0
NUMAGENTS      = 2
NUMSTEPSTOUPDATE = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer():
    def __init__(self, actionSize, bufferSize, batchSize, seed):
        self.actionSize = actionSize
        self.memory = deque(maxlen = bufferSize)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "nextState", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, nextState, done, numAgents):
        for i in range(numAgents):
            exp = self.experience(state[i], action[i], reward[i], nextState[i], done[i])
            self.memory.append(exp)
    
    def sample(self):
        experiences = random.sample(self.memory, k = self.batchSize)

        states     = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions    = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).float().to(device)
        rewards    = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        nextStates = torch.from_numpy(np.vstack([exp.nextState for exp in experiences if exp is not None])).float().to(device)
        dones      = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, nextStates, dones)

    def __len__(self):
        return len(self.memory)


class OUNoise():
    def __init__(self, size, seed, mu = 0, theta = 0.15, sigma = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state

class Agent():
    def __init__(self, stateSize, actionSize, randomSeed):
        self.stateSize  = stateSize
        self.actionSize = actionSize
        self.seed       = random.seed(randomSeed)

        self.actorLocal     = Actor(stateSize, actionSize, randomSeed).to(device)
        self.actorTarget    = Actor(stateSize, actionSize, randomSeed).to(device)
        self.actorOptimizer = optim.Adam(self.actorLocal.parameters(), lr = ACTORLEARNING)
        self.criticLocal     = Critic(stateSize, actionSize, randomSeed).to(device)
        self.criticTarget    = Critic(stateSize, actionSize, randomSeed).to(device)
        self.criticOptimizer = optim.Adam(self.criticLocal.parameters(), lr = CRITICLEARNING, weight_decay = WEIGHTDECAY)

        self.noise  = [OUNoise(actionSize, randomSeed) for i in range(NUMAGENTS)]
        self.memory = ReplayBuffer(actionSize, BUFFERSIZE, BATCHSIZE, randomSeed)

    def step(self, state, action, reward, nextState, done):
        if len(self.memory) > BATCHSIZE:
            for i in range(NUMSTEPSTOUPDATE):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, addNoise = True):
        state = torch.from_numpy(state).float().to(device)
        self.actorLocal.eval()
        with torch.no_grad():
            action = self.actorLocal(state).cpu().data.numpy()
        self.actorLocal.train()
        if addNoise:
            for i in range(NUMAGENTS):
                singleAction = action[i]
                for j in singleAction:
                    j += self.noise[i].sample()
        
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        states, actions, rewards, nextStates, dones = experiences

        actionsPred = self.actorLocal(states)
        actorLoss = -self.criticLocal(states, actionsPred).mean()
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()

        actionsNext = self.actorTarget(nextStates)
        QTargetNext = self.criticTarget(nextStates, actionsNext)
        QTargets = rewards + (gamma * QTargetNext * (1 - dones))
        QExpected = self.criticLocal(states, actions)
        criticLoss = F.mse_loss(QExpected, QTargets)
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.criticLocal.parameters(), 1)
        self.criticOptimizer.step()

        self.softUpdate(self.criticLocal, self.criticTarget, TAU)
        self.softUpdate(self.actorLocal, self.actorTarget, TAU)

    def softUpdate(self, localModel, targetModel, tau):
        for targetParam, localParam in zip(targetModel.parameters(), localModel.parameters()):
            targetParam.data.copy_(tau * localParam.data + (1.0 - tau) * targetParam.data)

    def saveExpInBuffer(self, state, action, reward, nextState, done):
        self.memory.add(state, action, reward, nextState, done, NUMAGENTS)    

    def reset(self):
        for i in range(NUMAGENTS):
            self.noise[i].reset()
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, cuda_index=1):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = th.device(
            f"cuda:{cuda_index}" if th.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=5e-4,
        max_q_target_iter=300,
        cuda_index=1,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        # Remove once you finish debugging
        self.n_actions = n_actions
        #
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.target_cntr = 0
        self.C = max_q_target_iter

        self.Q_pred = DeepQNetwork(
            lr=self.lr,
            input_dims=input_dims,
            fc1_dims=256,
            fc2_dims=256,
            n_actions=n_actions,
            cuda_index=cuda_index,
        )
        self.Q_target = self.Q_pred

        # it is very important to assign data types when working with PyTorch as it
        # enforces some form of type checking
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transitions(self, state, action, reward, state_, done):
        index = (
            self.mem_cntr % self.mem_size
        )  # Here we are using the % operator to see if we passed
        # the allowed memory size or not. If we did, then we will start from zero, and hence rewriting
        # the memory all over again
        # we could have simply used a queue instead of named arrays but it is simpler this way
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        self.target_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = th.tensor([observation]).to(self.Q_pred.device)
            actions = self.Q_pred.forward(state).cpu().data.numpy()
            # print(type(actions))
            action = np.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_pred.optimizer.zero_grad()  # This is particular to PyTorch
        batch_selection_max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(
            batch_selection_max_mem, self.batch_size, replace=False
        )  # It is important to make replace=False so that once you select an experience, you do not again
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = th.tensor(self.state_memory[batch]).to(self.Q_pred.device)
        new_state_batch = th.tensor(self.new_state_memory[batch]).to(self.Q_pred.device)
        reward_batch = th.tensor(self.reward_memory[batch]).to(self.Q_pred.device)
        terminal_batch = th.tensor(self.terminal_memory[batch]).to(self.Q_pred.device)

        action_batch = self.action_memory[batch]

        q_pred = self.Q_pred.forward(state_batch)[batch_index, action_batch]
        ############ This Part needs to be changed to be constant for C iterations ################
        if self.target_cntr > self.C:
            self.target_cntr = 0
            self.Q_target = self.Q_pred
        q_next = self.Q_target.forward(new_state_batch)
        # print(q_next.shape, f"[{self.batch_size},{self.n_actions}]")
        # print(terminal_batch.shape)
        q_next[terminal_batch] = 0
        ###########################################################################################
        q_target = reward_batch + self.gamma * th.max(q_next, dim=1)[0]

        loss = self.Q_pred.loss(q_target, q_pred).to(self.Q_pred.device)
        loss.backward()
        self.Q_pred.optimizer.step()

        self.epsilon = (
            (self.epsilon - self.eps_dec)
            if self.epsilon > self.eps_min
            else self.eps_min
        )

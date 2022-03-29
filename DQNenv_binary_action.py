# from NSMQV.NSMQV_Simulation import NSMQV_Simulation
from NSMQV.NSMQV_toEdge import NSMQV_Simulation
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
import logging

# Cheating mode speeds up the training process
CHEAT = True


# Basic Q-netowrk
class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # Two fully-connected layers, input (state) to hidden & hidden to output (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# Deep Q-Network, composed of one eval network, one target network
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter,
                 memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)

        # initialize memory, each memory slot is of size (state + next state + reward + action)
        ''' NSMQV_toEdge '''
        self.memory = np.zeros((memory_capacity, n_states * 2 + 3))  # 根據需要存的量去調整 size
        ''' NSMQV_Simulation '''
        # self.memory = np.zeros((memory_capacity, n_states * 2 + 4))  # 根據需要存的量去調整 size

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0  # for target network update

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def choose_action(self, state):

        # while True:
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon:  # random
            # action = np.random.rand(6)
            # action = np.random.randint(0, self.n_actions)

            ''' NSMQV_toEdge '''
            action = np.random.randint(self.n_actions, size=2)
            ''' NSMQV_Simulation '''
            # action = np.random.randint(self.n_actions / 3, size=3)
        else:  # greedy
            actions_value = self.eval_net(x)  # feed into eval net, get scores for each action

            ''' NSMQV_toEdge '''
            # action = torch.max(actions_value, 0)[1]
            action_max = torch.max(actions_value, 1)[1].data.numpy()[0]  # choose the one with the largest score
            action_min = torch.min(actions_value, 1)[1].data.numpy()[0]  # choose the one with the smallest score
            action = [action_max, action_min]
            ''' NSMQV_Simulation '''
            # actions_value = torch.reshape(actions_value, (-1,))  # reshape to single dimension
            # actions_value_x, actions_value_y, actions_value_z = torch.split(actions_value, [6, 6, 6])  # cut the value [UE, E]
            # actions_value_x = torch.unsqueeze(torch.FloatTensor(actions_value_x), 0)  # reshape the tensor
            # actions_value_y = torch.unsqueeze(torch.FloatTensor(actions_value_y), 0)
            # actions_value_z = torch.unsqueeze(torch.FloatTensor(actions_value_z), 0)
            # action_x = torch.max(actions_value_x, 1)[1].data.numpy()[0]  # 挑選最高分的 action
            # action_y = torch.max(actions_value_y, 1)[1].data.numpy()[0]
            # action_z = torch.max(actions_value_z, 1)[1].data.numpy()[0]
            # action = [action_x, action_y, action_z]  # [UE, E] 的動作

            # action_sum = 0
            # for a in range(action.size):
            #     action_sum += action[a]
            #
            # if action_sum <= 0:
            #     break

            """
            Actions:
                Type: Discrete(6)
                Num   Action
                0     [+u, -u, 0]
                1     [+u, 0, -u]
                2     [0, +u, -u]
                3     [0, -u, +u]
                4     [-u, +u, 0]
                5     [-u, 0, +u]

                Num   Action
                0     [+u, 0, -u]
                1     [+u, -u, 0]
                2     [0, +u, -u]
                3     [-u, +u, 0]
                4     [0, -u, +u]
                5     [-u, 0, +u]
            """

        return action

    def store_transition(self, state, action, reward, next_state):
        # Pack the experience
        transition = np.hstack((state, action, reward, next_state))

        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Randomly select a batch of memory to learn from
        if self.memory_counter < self.memory_capacity & self.memory_counter < self.batch_size:
            sample_index = np.random.choice(self.memory_counter, self.memory_counter)
        elif self.memory_counter < self.memory_capacity & self.memory_counter > self.batch_size:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        ''' NSMQV_toEdge '''
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 2].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states + 2:self.n_states + 3])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])
        ''' NSMQV_Simulation '''

        # Compute loss between Q values of eval net & target net
        q_eval = self.eval_net(b_state).gather(1,
                                               b_action)  # evaluate the Q values of the experiences, given the states & actions taken at that time
        q_next = self.target_net(b_next_state).detach()  # detach from graph, don't backpropagate
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # compute the target Q values
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every few iterations (target_replace_iter), i.e. replace target net with eval net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


if __name__ == "__main__":
    ''' UE E EU '''
    '''
    env = NSMQV_Simulation()
    # path = 'record.txt'

    # Environment parameters
    n_actions = 18
    n_states = 9

    # Hyper parameters
    n_hidden = 50
    batch_size = 300
    lr = 0.01  # learning rate
    epsilon = 0.1  # epsilon-greedy, factor to explore randomly
    gamma = 0.9  # reward discount factor
    target_replace_iter = 100  # target network update frequency
    memory_capacity = 10000  # 多久 train 一次
    n_episodes = 100  # if CHEAT else 4000

    # Create DQN
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

    # alloc_UE = [1.0849649711956881E7, 1.5998360758527812E7, 2.1151989529515307E7]  # 上傳頻寬的服務率(傳輸率)
    # alloc_E = [4693948.052696463, 29122.49841629079, 76929.44888724666]  # 在Edge的服務率(傳輸率)
    # alloc_EU = [2.44036944403482E7, 2.0173945508827038E7, 3.842236005082476E7]  # 下載頻寬的服務率(傳輸率)

    alloc_UE = [40000000 / 3, 40000000 / 3, 40000000 / 3]  # 上傳頻寬的服務率(傳輸率)
    alloc_E = [4000000 / 3, 4000000 / 3, 4000000 / 3]  # 在Edge的服務率(傳輸率)
    alloc_EU = [63000000/3, 63000000/3, 63000000/3]  # 下載頻寬的服務率(傳輸率)

    # Collect experience
    for i_episode in range(n_episodes):
        t = 0  # timestep
        rewards = 0  # accumulate rewards for each episode

        state = alloc_UE + alloc_E + alloc_EU  # reset environment to initial state for each episode
        action = alloc_UE + alloc_E + alloc_EU
        old_reward = 1
        while True:
            env.output_QV(action)

            # Agent takes action
            action_type = dqn.choose_action(state)  # choose an action based on DQN

            # next_state: 調整變量；new_reward: QoS違反機率
            next_state, new_reward, done = env.step(action_type)  # do the action, get the reward
            action += next_state

            # if 分配量小於0 , 恢復原本的分配量
            for i in range(len(action)):
                if action[i] < 0:
                    # 針對小於0的資源，恢復原本的值(一次調整三種服務)
                    if i < 3:
                        action[:3] -= next_state[:3]
                    elif 2 < i < 6:
                        action[3:6] -= next_state[3:6]
                    else:
                        action[6:] -= next_state[6:]

            # # Cheating part: modify the reward to speed up training process
            # if CHEAT:
            #     x, v, theta, omega = next_state
            #     r1 = (env.x_threshold - abs(
            #         x)) / env.x_threshold - 0.8  # reward 1: the closer the cart is to the center, the better
            #     r2 = (env.theta_threshold_radians - abs(
            #         theta)) / env.theta_threshold_radians - 0.5  # reward 2: the closer the pole is to the center, the better
            #     reward = r1 + r2

            # Keep the experience in memory
            dqn.store_transition(state, action_type, 1 - new_reward, action)

            # Accumulate reward
            # 如果qos變好就reward+1，變差就reward-1
            # if new_reward < old_reward:
            #     rewards += 1
            # else:
            #     rewards -= 1
            # 如果qos接近滿足率100%(大於0.5)就reward+qos，遠離(小於0.5)就reward-qos
            if new_reward < 0.5:
                if new_reward <= old_reward:
                    rewards += 1 - new_reward
                else:
                    rewards -= new_reward
            else:
                rewards -= 1
            # 如果qos接近滿足率100%(大於0.5)就reward+1，遠離(小於0.5)就reward-1
            # if new_reward < 0.5:
            #     rewards += 1
            # else:
            #     rewards -= 1
            # 如果qos變好就reward+qos，變差就reward-qos !!!!!效果很差!!!!!
            # if new_reward < old_reward:
            #     rewards += 1 - new_reward
            # else:
            #     rewards -= 1 - new_reward
            # 如果qos變好就reward+qos，變差就reward-qos !!!!!效果很差!!!!!
            # if new_reward < old_reward:
            #     rewards += 1 - new_reward
            # else:
            #     rewards -= new_reward

            old_reward = new_reward

            print('Episode: {} , Total Rewards: {} , QoS violation: {} , Action type: {}'.format(t + 1, rewards, new_reward, action_type))
            # print('Action type: ', action_type)
            print('Action: ', state)

            f = open('record.txt', 'a')
            f.write('Episode: {} , Rewards: {} , QoS violation: {} , Action type: {} \n'.format(t+1, rewards, new_reward, action_type))
            f.close()

            # If enough memory stored, agent learns from them via Q-learning
            # if dqn.memory_counter > memory_capacity:
            dqn.learn()

            # Transition to next state
            state = action

            if done:
                print('!!!!!!!! Done !!!!!!!!')
                print('Episode finished after {} timesteps, total rewards {}'.format(t + 1, rewards))
                print('!!!!!!!!!!!!!!!!!!!!!!')
                break

            t += 1

    # env.close()
    '''

    ''' only UE '''

    env = NSMQV_Simulation()
    # path = 'record.txt'

    # Environment parameters
    n_actions = 3
    n_states = 3

    # Hyper parameters
    n_hidden = 50
    batch_size = 300
    lr = 0.01  # learning rate
    epsilon = 0.1  # epsilon-greedy, factor to explore randomly
    gamma = 0.9  # reward discount factor
    target_replace_iter = 100  # target network update frequency
    memory_capacity = 10000  # 多久 train 一次
    n_episodes = 100  # if CHEAT else 4000

    # Create DQN
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

    # alloc_UE = [1.0849649711956881E7, 1.5998360758527812E7, 2.1151989529515307E7]  # 上傳頻寬的服務率(傳輸率)
    alloc_UE = [40000000 / 3, 40000000 / 3, 40000000 / 3]  # 上傳頻寬的服務率(傳輸率)

    # Collect experience
    for i_episode in range(n_episodes):
        t = 0  # timestep
        rewards = 0  # accumulate rewards for each episode

        state = alloc_UE  # reset environment to initial state for each episode
        action = alloc_UE
        old_reward = 1
        while True:
            env.output_QV(action)

            # Agent takes action: action = [action_max, action_min]
            action_type = dqn.choose_action(state)  # choose an action based on DQN

            # next_state: 調整變量；new_reward: QoS違反機率
            next_state, new_reward, done = env.step(action_type)  # do the action, get the reward
            action += next_state

            # if 分配量小於0 , 恢復原本的分配量
            for i in range(len(action)):
                if action[i] < 0:
                    # 針對小於0的資源，恢復原本的值(一次調整三種服務)
                    action -= next_state

            # # Cheating part: modify the reward to speed up training process
            # if CHEAT:
            #     x, v, theta, omega = next_state
            #     r1 = (env.x_threshold - abs(
            #         x)) / env.x_threshold - 0.8  # reward 1: the closer the cart is to the center, the better
            #     r2 = (env.theta_threshold_radians - abs(
            #         theta)) / env.theta_threshold_radians - 0.5  # reward 2: the closer the pole is to the center, the better
            #     reward = r1 + r2

            # Keep the experience in memory
            dqn.store_transition(state, action_type, 1 - new_reward, action)

            # Accumulate reward
            # 如果qos變好就reward+1，變差就reward-1
            # if new_reward < old_reward:
            #     rewards += 1
            # else:
            #     rewards -= 1
            # 如果qos接近滿足率100%(大於0.5)就reward+qos，遠離(小於0.5)就reward-qos
            if new_reward < 0.5:
                if new_reward <= old_reward:
                    rewards += 1 - new_reward
                else:
                    rewards -= new_reward
            else:
                rewards -= 1
            # 如果qos接近滿足率100%(大於0.5)就reward+1，遠離(小於0.5)就reward-1
            # if new_reward < 0.5:
            #     rewards += 1
            # else:
            #     rewards -= 1
            # 如果qos變好就reward+qos，變差就reward-qos !!!!!效果很差!!!!!
            # if new_reward < old_reward:
            #     rewards += 1 - new_reward
            # else:
            #     rewards -= 1 - new_reward
            # 如果qos變好就reward+qos，變差就reward-qos !!!!!效果很差!!!!!
            # if new_reward < old_reward:
            #     rewards += 1 - new_reward
            # else:
            #     rewards -= new_reward

            old_reward = new_reward

            print('Episode: {} , Total Rewards: {} , QoS violation: {} , Action type: {}'.format(t + 1, rewards,
                                                                                                 new_reward,
                                                                                                 action_type))
            # print('Action type: ', action_type)
            print('Action: ', state)

            f = open('record.txt', 'a')
            f.write(
                'Episode: {} , Rewards: {} , QoS violation: {} , Action type: {} \n'.format(t + 1, rewards, new_reward,
                                                                                            action_type))
            f.close()

            # If enough memory stored, agent learns from them via Q-learning
            # if dqn.memory_counter > memory_capacity:
            dqn.learn()

            # Transition to next state
            state = action

            # if done:
            #     print('!!!!!!!! Done !!!!!!!!')
            #     print('Episode finished after {} timesteps, total rewards {}'.format(t + 1, rewards))
            #     print('!!!!!!!!!!!!!!!!!!!!!!')
            #     break

            t += 1

    # env.close()


# File: agent_brain.py
# Description: Creating brain for the agent based on the Sarsa algorithm
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing function from the env.py
from env import final_states


# 定义Sarsa算法类
class SarsaTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # 动作集合
        self.actions = actions
        # 学习率α
        self.lr = learning_rate
        # 折扣因子 gamma
        self.gamma = reward_decay
        # 贪婪策略参数 epsilon
        self.epsilon = e_greedy
        # 创建完整Q表
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # 创建最终路径Q表
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

        # 动作选择函数

    def choose_action(self, observation):
        # 检查状态是否在表中存在
        self.check_state_exist(observation)

        # ε-贪婪策略 90%概率选择当前Q值最大动作
        if np.random.uniform() < self.epsilon:

            # 提取当前状态下所有动作的价值函数
            state_action = self.q_table.loc[observation, :]

            # 打乱顺序，避免每次选择的动作都为序号偏前的动作（防止Q值最大的动作有多个）
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()  # 返回第一个遇到的最大Q值
        else:
            # 10%概率随机探索
            action = np.random.choice(self.actions)
        return action

    # 学习和更新函数
    def learn(self, state, action, reward, next_state, next_action):
        # 检查下一个状态，如果不在Q-table中则将其加入到Q-table中
        self.check_state_exist(next_state)

        # 获取当前的Q值（预测值） 即目前Q_table内存储的Q值
        q_predict = self.q_table.loc[state, action]

        # 检测下一个状态是空地还是目标点还是障碍物
        if next_state != 'goal' and next_state != 'obstacle':

            # 更新q_target使用实际选择的下一个动作
            q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]
        else:
            q_target = reward  # 到goal +100 到达障碍-100 在环境中已配置好

        # 用得到的新情况更新Q表
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        return self.q_table.loc[state, action]

    # 检查状态（如[5.0, 40.0]）是否在Q表中，如果不在就把该状态加入Q表中
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series(
                    [0] * len(self.actions),  # 创建一个全0列表，长度等于动作数量
                    index=self.q_table.columns,  # 列名作为series的索引
                    name=state,  # 状态名作为series的名称
                )
            )

    # Printing the Q-table with states
    def print_q_table(self):
        # 获取最终成功路径的坐标列表
        e = final_states()

        # Comparing the indexes with coordinates and writing in the new Q-table values
        for i in range(len(e)):
            state = str(e[i])  # state = '[5.0, 40.0]'
            # Going through all indexes and checking
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)

    # Plotting the results for the number of steps
    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
        plt.show()


# Creating class for the Q-learning table
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # List of actions
        self.actions = actions
        # Learning rate
        self.lr = learning_rate
        # Value of gamma
        self.gamma = reward_decay
        # Value of epsilon
        self.epsilon = e_greedy
        # Creating full Q-table for all cells
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # Creating Q-table for cells of the final route
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # Function for choosing the action for the agent
    def choose_action(self, observation):
        # Checking if the state exists in the table
        self.check_state_exist(observation)
        # Selection of the action - 90 % according to the epsilon == 0.9
        # Choosing the best action
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]  # 提取当前状态下所有动作的价值函数
            # print(state_action)
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # 打乱顺序，避免每次选择的动作都为序号偏前的动作
            # print(state_action)
            action = state_action.idxmax()
        else:
            # Choosing random action - left 10 % for choosing randomly
            action = np.random.choice(self.actions)
        return action

    # Function for learning and updating Q-table with new knowledge
    def learn(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table 如果不在Q-table中则将其加入到Q-table中
        self.check_state_exist(next_state)

        # Current state in the current position
        q_predict = self.q_table.loc[state, action]  # 预测的Q值,即目前Q_table内存储的Q值

        # Checking if the next state is free or it is obstacle or goal
        if next_state != 'goal' or next_state != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()  # 实际最大值 由动作奖励以及下一状态的最大Q值×折损率组成
        else:
            q_target = reward

        # Updating Q-table with new knowledge
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)  # 更新Q值

        return self.q_table.loc[state, action]

    # Adding to the Q-table new states
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # Printing the Q-table with states
    def print_q_table(self):
        # Getting the coordinates of final route from env.py
        e = final_states()

        # Comparing the indexes with coordinates and writing in the new Q-table values
        for i in range(len(e)):
            state = str(e[i])  # state = '[5.0, 40.0]'
            # Going through all indexes and checking
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)

    # Plotting the results for the number of steps
    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
        plt.show()
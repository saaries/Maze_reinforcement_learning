import gym
import gym_maze
import matplotlib.pyplot as plt
import numpy as np
import random
import os

maze_size = 50
action_num = 4
EPISODES = 2000

maze_name = "maze-sample-"+str(maze_size)+"x"+str(maze_size)+"-v0"
class QLearning():
    def __init__(self, state_list, action_size):
        self.state_list = state_list
        self.action_size = action_size
        self.gamma = 0.95
        # e-贪婪
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.alpha = 0.7
        self.q = self.init_q()

    def init_q(self):
        q = {}
        for i in self.state_list:
            q[i] = {}
            for j in range(self.action_size):
                q[i][j] = 0
        return q

    # 动作选择策略
    def choose_act(self, state):
        # e-贪婪：有 epsilon 的概率选择随机操作
        if random.random() < self.epsilon:
            return np.random.randint(action_num)

        # e-贪婪：有 1-epsilon 的概率选择Q(s, a)值最大的操作
        val = -10000000
        action = 0
        for i in range(self.action_size):
            if self.q[state][i] > val:
                action = i
                val = self.q[state][i]
        return action

    # 更新 Q 表
    def update_q(self, state, action, reward, state_):
        val = -10000000
        for i in range(self.action_size):
            if self.q[state_][i] > val:
                val = self.q[state_][i]
        self.q[state][action] = self.q[state][action] + \
            self.alpha * (reward + self.gamma * val - self.q[state][action])

class SARSA():
    def __init__(self, state_list, action_size):
        self.state_list = state_list
        self.action_size = action_size
        self.gamma = 0.95
        # e-贪婪
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.alpha = 0.7
        self.q = self.init_q()

    def init_q(self):
        q = {}
        for i in self.state_list:
            q[i] = {}
            for j in range(self.action_size):
                q[i][j] = 0
        return q

    # 动作选择策略
    def choose_act(self, state):
        # e-贪婪：有 epsilon 的概率选择随机操作
        if random.random() < self.epsilon:
            return np.random.randint(action_num)

        # e-贪婪：有 1-epsilon 的概率选择Q(s, a)值最大的操作
        val = -10000000
        action = 0
        for i in range(self.action_size):
            if self.q[state][i] > val:
                action = i
                val = self.q[state][i]

        return action

    # 更新 Q 表
    def update_q(self, state, action, reward, state_, action_):
        self.q[state][action] = self.q[state][action] + \
            self.alpha * (reward + self.gamma * self.q[state_][action_] - self.q[state][action])

def Q():
    # 初始化环境
    env = gym.make(maze_name)
    state_list = []

    # 初始化 Q 表
    for i in range(maze_size):
        for j in range(maze_size):
            state_list.append((i, j))

    # 记录每次训练所用的时间步
    step_used = []

    # Q-Learning 算法实现：agent
    agent = QLearning(state_list, action_num)

    # 尝试 500 回合
    for episode in range(EPISODES):
        # 重置迷宫环境，state = (0,0)
        state = env.reset()
        state = (state[0], state[1])

        for step in range(1000):
            # env.render()
            # 依策略选择动作
            action = agent.choose_act(state)
            state_, reward, done, _ = env.step(action)
            state_ = (state_[0], state_[1])
            # 每个时间步更新一次 Q 表
            agent.update_q(state, action, reward, state_)
            # S←S'
            state = state_
            # 当使用步数>499，重新开始回合
            if done or step == 999:
                if step < 999:
                    agent.epsilon = 0
                    # e-贪婪：不断减小 epsilon 值，加速收敛
                else:
                    agent.epsilon *= agent.epsilon_decay

                print("episode: {}/{}, Took = {} steps, Epsilon = {}"
                      .format(episode, EPISODES, step, agent.epsilon))

                step_used.append(step)
                break

    return step_used

def S():
    # 初始化环境
    env = gym.make(maze_name)
    state_list = []

    # 初始化 Q 表
    for i in range(maze_size):
        for j in range(maze_size):
            state_list.append((i, j))

    # 记录每次训练所用的时间步
    step_used = []

    # SARSA 算法实现：agent
    agent = SARSA(state_list, action_num)

    # 尝试 500 回合
    for episode in range(EPISODES):
        # 重置迷宫环境，state = (0,0)
        state = env.reset()
        state = (state[0], state[1])
        # 依策略选择动作
        action = agent.choose_act(state)

        for step in range(1000):
            # env.render()
            state_, reward, done, _ = env.step(action)
            state_ = (state_[0], state_[1])
            action_ = agent.choose_act(state_)
            # 每个时间步更新一次 Q 表
            agent.update_q(state, action, reward, state_, action_)
            # S←S', A←A'
            state = state_
            action = action_
            # 当使用步数>499，重新开始回合
            if done or step == 999:
                if step < 999:
                    agent.epsilon = 0
                    # e-贪婪：不断减小 epsilon 值，加速收敛
                else:
                    agent.epsilon *= agent.epsilon_decay

                print("episode: {}/{}, Took = {} steps, Epsilon = {}"
                      .format(episode, EPISODES, step, agent.epsilon))
                step_used.append(step)

                break

    return step_used


if __name__ == '__main__':
    Qstep_used = Q()
    print()
    Sstep_used = S()

    plt.plot(Qstep_used, linewidth='0.8', label="Q-Learning", color='red')
    plt.plot(Sstep_used, linewidth='0.8', label="SARSA", color='blue')

    plt.xlabel('Episode')
    plt.ylabel('Step used')

    plt.legend(loc='upper right')

    path = 'result'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + '/size'+str(maze_size)+'.png')
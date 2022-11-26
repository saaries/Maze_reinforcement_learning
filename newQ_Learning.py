import gym
import gym_maze
import matplotlib.pyplot as plt
import numpy as np
import random
import os

maze_size = 10
action_num = 4
EPISODES = 10000

class QLearning():
    def __init__(self, state_list, action_size):
        self.state_list = state_list
        self.action_size = action_size
        self.gamma = 0.95
        # e-贪婪
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.alpha = 0.5
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

if __name__ == '__main__':
    # 初始化环境
    str_env_name="maze-random-"+str(maze_size)+"x"+str(maze_size)+"-v0"
    env = gym.make(str_env_name)
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
        # if episode > 10:
        #     agent.epsilon = 0

        # 重置迷宫环境，state = (0,0)
        state, _ = env.reset()
        state = (state[0], state[1])
        
        for step in range(10000):
            env.render()
            # 依策略选择动作
            action = agent.choose_act(state)
            state_, reward, done, _, _ = env.step(action)
            state_ = (state_[0], state_[1])
            # 每个时间步更新一次 Q 表
            agent.update_q(state, action, reward, state_)
            # S←S'
            state = state_
            # 当使用步数>499，重新开始回合
            if done or step == 9999:
                print("episode: {}/{}, Took = {} steps, Epsilon = {}"
                      .format(episode, EPISODES, step, agent.epsilon))
                step_used.append(step)

                # e-贪婪：不断减小 epsilon 值，加速收敛
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

                break


    plt.plot(step_used, 'r')
    path = 'result'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + '/qlearning.png')
import gym
import gym_maze
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from queue import Queue

maze_size = 30
action_num = 4
endFound = 0
s = ""
que = []
path={}

def dfs(x1, y1, adjList, x2, y2, discovered):
    global endFound
    global s
    if endFound == 0:
        discovered[x1][y1] = 1
        for i in adjList[x1][y1]:
            s += i
            if i == 'l':
                if x1 == x2 and y1-1 == y2:
                    endFound = 1
                    return
                if y1-1 >= 0 and discovered[x1][y1-1] == 0:
                    dfs(x1, y1-1, adjList, x2, y2, discovered)

            if i == 'r':
                if x1 == x2 and y1+1 == y2:
                    endFound = 1
                    return
                if y1+1 < maze_size and discovered[x1][y1+1] == 0:
                    dfs(x1, y1+1, adjList, x2, y2, discovered)

            if i == 'u':
                if x1-1 == x2 and y1 == y2:
                    endFound = 1
                    return

                if x1-1 >= 0 and discovered[x1-1][y1] == 0:
                    dfs(x1-1, y1, adjList, x2, y2, discovered)

            if i == 'd':
                if x1+1 == x2 and y1 == y2:
                    endFound = 1
                    return
                if x1+1< maze_size and discovered[x1+1][y1] == 0:
                    dfs(x1+1, y1, adjList, x2, y2, discovered)

            if endFound == 0:
                s = s[:-1]
            else:
                return


# def bfs(x1, y1, adjList, x2, y2, discovered):
#     global endFound
#     global path
#     global s
#     path[(x1, y1)] = None
#     discovered[x1][y1] = 1
#
#     x_cord = [0, 1, 0, -1]
#     y_cord = [1, 0, -1, 0]
#
#     que.append([x1, y1])
#     while que:
#         d = que.pop(0)
#
#         for i in range(0, 4):
#             x_temp = x_cord[i] + d[0]
#             y_temp = y_cord[i] + d[1]
#             if x_temp == x2 and y_temp == y2:
#                 path[(x_temp, y_temp)] = (d[0], d[1])
#
#             if (x_temp >= 0 and x_temp < maze_size and y_temp >= 0 and y_temp < maze_size
#                     and discovered[x_temp][y_temp] == 0):
#                 discovered[x_temp][y_temp]=1
#                 path[(x_temp, y_temp)] = (d[0], d[1])
#                 que.append([x_temp,y_temp])
#
#             if i == 'd':
#                 if x1 + 1 == x2 and y1 == y2:
#                     endFound = 1
#                     path[(x1+1, y1)] = (d[0], d[1])
#                     return
#                 if x1 + 1 < maze_size and discovered[x1 + 1][y1] == 0:
#                     discovered[x1 + 1][y1] = 1
#                     path[(x1, y1+1)] = (d[0], d[1])
#                     s += i
#                     que.put([x1 + 1, y1])



def getPath(x, y):
    pathCount=0
    while (path[(x, y)] != None):
        pathCount = pathCount + 1
        x, y = path[(x, y)]
    print(pathCount)
    return pathCount

if __name__ == '__main__':
    # 初始化环境
    env = gym.make("maze-random-30x30-v0")
    compass = ['u','d','r','l']
    maze_node_list = []
    observation = []
    action_space = []
    reshape=[]
    acts=''

    for i in range(maze_size):
        for j in range(maze_size):
            observation.append([j, i])

    for ob in observation:
        for action_index in range(4):
            if env.unwrapped.mazemap(ob, action_index):
                acts += compass[action_index]
        action_space.append(acts)
        acts=''

    for i in range(maze_size):
        for j in range(maze_size):
            reshape.append(action_space[i*maze_size + j])

        maze_node_list.append(reshape)

        reshape=[]

    discovered_node_list = [[0 for col in range(maze_size)] for row in range(maze_size)]
    dfs(0, 0, maze_node_list, maze_size-1, maze_size-1, discovered_node_list)
    # getPath(8, 9)
    # for _ in path:
    print(s)

    while 1:
        env.render()

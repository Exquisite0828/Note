"""
RRT_2D
@author: huiming zhou
"""

from env import Environment
from env import final_states
from plotting import Plotting
from agent_brain import SarsaTable


def update():
    # 记录每个episode（回合）所用的步数
    steps = []

    # 记录每个episode的总成本
    all_costs = []

    # 进行多次学习过程，每一次episode都是一个从起点到终点完成的过程
    for episode in range(10000):
        # Initial Observation
        # 将机器人放在（0，0）处并清空路径字典
        observation = env.reset()

        # 更新当前episode的步数计数器
        i = 0

        # 更新当前episode的累计成本
        cost = 0

        # 寻找动作的依据为以一定概率选择目前状态下动作值函数最大的动作
        # 以一定概率随机选择（随机选择的目的是增加探索率）
        action = RL.choose_action(str(observation))

        # 重复执行直到达到目标节点或障碍物
        while True:
            # 将该动作执行，得到奖励值，下个状态以及是否结束寻路标志
            observation_, reward, done = env.step(action)

            action_ = RL.choose_action(str(observation_))
            # 计算整个过程中的cost，与算法无关，用作后续查看算法执行情况

            # 不过在learn函数中完成了Q_table的更新
            cost += RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            i += 1

            if done:
                steps += [i]
                all_costs += [cost]
                break


    env.final()

    RL.print_q_table()

    RL.plot_results(steps, all_costs)



if __name__ == "__main__":
    x_start = (5, 5)  # Starting node
    x_goal = (75, 35)  # Goal node

    env = Environment(x_start,x_goal)
    RL = SarsaTable(actions=list(range(env.n_actions)),
                    learning_rate=0.1,
                    reward_decay=0.93,
                    e_greedy=0.9) #初始化Sarsa_table

    # rrt = Rrt(x_start, x_goal, 0.5, 0.05, 10000)
    #path = rrt.planning()
    update()
    plotting = Plotting(x_start, x_goal,env)
    path=list(final_states().values())
    path.insert(0,x_start)
    if path:
        print("我要画图了")
        plotting.animation([], path, "Sarsa", True)
    else:
        print("No Path Found!")


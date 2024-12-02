# time 20230325
# by qian
# 利用ddpg 做出缓存大时间尺度上，在小时间尺度上，ddpg算法带宽、计算资源分配和计算卸载比例的决策！！
import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from arguments import parse_args
from Replay_buffer import ReplayBuffer
from model import DDPG
from singleMEC1.env import MultiAgentEnv
from singleMEC1.my_world import Scenario
from collections import namedtuple

def make_off_env():
    # make the environment for offload & bandwidth allocation & caching
    scenario = Scenario()
    print("create the env ")
    world = scenario.make_world()
    # def __init__(self, world, reset_callback_small=None ,rest_callback_cache=None,observation_callback = None, reward = None ):
    env = MultiAgentEnv(world,scenario.reset_world,scenario.reset_cache, scenario.observation_total,scenario.reward)
    print("end make env")
    return env


def train(arglist):
    # train the agents in the offload environment
    # input arglist: env parameters
    # out the reward of agents , the model needed to save
    print("""step1: create the environment """)
    env = make_off_env()
    print('step 1 Env {} is right ...'.format(arglist.scenario_name))

    print("""step2: create agents""")
    # 首先是获取到 服务器智能体的状态和动作的维度，用来产生智能体：
    #print("env.observation_space_server_cache",env.observation_space_server_cache[0])
    obs_shape_server = env.observation_space[0].shape[0]
    print('obs_shape_n_server',obs_shape_server)
    action_shape_server_cache = env.action_space_server_cache[0].shape[0]
    print('action_shape_n_server', action_shape_server_cache)
    action_shape_small = env.action_space_small[0].shape[0]
    print('action_shape_n_server', action_shape_small)
    # ddpg
    agent_server_cache = DDPG(obs_shape_server,action_shape_server_cache, arglist)
    # small agent network
    agent_small = DDPG(obs_shape_server + action_shape_server_cache, action_shape_small ,arglist)

    print('step 2 The {} agents are inited ...'.format(1))

    print('step 3 starting iterations ...')
    cache_fre = 1
    game_step = 0
    game_step_cache = 0
    # 每一步用户和服务器的效用
    users_time = []
    server_switch_cost = []
    rewards_server_cache = []
    reward_small_ra = []

    # 刚开始的时候。是上下两层都进行重置！！！只用reset_high 既可
    # 获得到初始化的状态
    # max_epsode 500
    # each_episode: 100 step, each 5step 做一下缓存， 100次 此的卸载和带宽分配决策
    var_cache = 1
    var_small = 1
    for episode_gone in range(arglist.max_episode):
        env.world.episode = episode_gone
        obs= env.reset_high()
        action_cache = 0
        reward_cache = 0
        action_small = 0
        reward_small = 0

        ep_cache_cost =0.0
        ep_reward_cache =0.0
        ep_reward_small =0.0
        ep_time = 0.0
        reward_average_small = []
        action_server_cache = []
        cache_value = 0
        for step in range(arglist.per_episode_max_len):  # 每一回合里面的step 20 * 5数目
            print("this is the step " + str(step) + " of episode " + str(episode_gone))
            env.world.game_step = game_step
            server = env.server
            if step % cache_fre == 0:
                var_cache *= 0.9882
                var_small *= 0.9995
                # 500 步0.988
                past_cache = server.cache
                #print("obsivation:",obs)
                action_server_cache = agent_server_cache.select_action(obs)
                #print("action_server_cache",action_server_cache)
                action_server_cache = np.clip(np.random.normal(action_server_cache, var_cache), -1, 1)
                #print("noise: action_server_cache", action_n_server_cache)
                env._set_action_server_cache(action_server_cache)
                #env._set_action_ramdom_cache(action_server_cache)
                current_cache = server.cache
                #print("current_cache", current_cache)
                #print("past_cache", past_cache)
                for i in range(len(current_cache)):
                    if current_cache[i] == 1 and past_cache[i] == 0:
                        communcation_cost = server.get_power * (env.task[i].get_cache_size / server.backhaul_rate)
                        cache_value += communcation_cost / (cache_fre * 20)
                #env._set_action_ramdom_cache(action_n_server_cache)
                # offloading ,computing frequence and bandwidth 决策
                #print("obsivation:obs_small", obs_small)
                s = []
                for i in obs:
                    s.append(i)
                for i in server.cache:
                    s.append(i)
                s = np.array(s)
                action_small = agent_small.select_action(s)
                #print("action_small", action_small)
                action_small = np.clip(np.random.normal(action_small, var_small), -1, 1)
               # print("noise: action_small", action_small)
                env._set_action_small(action_small)
                #env._set_action_small_ave(action_small)
                #culculate the average delay of small steps
                reward_small, time = env._get_reward()
                constraint_over_cache_cap = 0
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.task[i].get_cache_size
                if cache_size > server.get_cache_cap:
                    constraint_over_cache_cap = (cache_size - server.get_cache_cap) / (1024 * 1024 * 1024 * 8)
                #print("constraint_over_cache_cap", constraint_over_cache_cap)
                #print("reward_average_small", reward_average_small, np.mean(reward_average_small))
                reward_new = reward_small - cache_value - constraint_over_cache_cap
                #print("the small reward :", reward_small)
                #print("cache_value",cache_value)
                reward_average_small.append(reward_small)

                #更新每个时隙里面用户的请求，大小
                env.world.step()
                next_obs_small = env._get_obs()
                s_ = []
                for i in next_obs_small:
                    s_.append(i)
                for i in server.cache:
                    s_.append(i)
                s_ = np.array(s_)
                # print("next_obs_server_n_cache",next_obs_server_n_cache)


                agent_small.replay_buffer.push((s, action_small,
                                                       s_, reward_new))
                obs = next_obs_small

                ep_reward_small += reward_new
                ep_time += time
                #obs_n_server_cache = next_obs_server_n_cache
                if cache_fre ==1:
                    agent_server_cache.replay_buffer.push((s, action_server_cache,
                                                           obs, reward_cache))
                    ep_reward_cache += reward_new
                    ep_cache_cost += cache_value
                    reward_average_small = []

                game_step += 1
                game_step_cache +=1

            else:
                #print("single step !!!")
                # offloading ,computing frequence and bandwidth 决策
                #print("obsivation:obs_small", obs)
                s = []
                for i in obs:
                    s.append(i)
                for i in server.cache:
                    s.append(i)
                s = np.array(s)
                action_small = agent_small.select_action(s)
                #print("action_small", action_small)
                action_small = np.clip(np.random.normal(action_small, var_small), -1, 1)
                #print("noise: action_small", action_small)
                env._set_action_small(action_small)
                #env._set_action_small_ave(action_small)
                reward_small, time = env._get_reward()
                #print("the small reward :", reward_small)
                reward_average_small.append(reward_small)
                constraint_over_cache_cap = 0
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.task[i].get_cache_size
                if cache_size > server.get_cache_cap:
                    constraint_over_cache_cap = (cache_size - server.get_cache_cap) / (1024 * 1024 * 1024 * 8)
                #print("constraint_over_cache_cap", constraint_over_cache_cap)
                #print("reward_average_small", reward_average_small, np.mean(reward_average_small))
                reward_new = reward_small - cache_value - constraint_over_cache_cap
                env.world.step()
                next_obs_small = env._get_obs()
                s_ = []
                for i in next_obs_small:
                    s_.append(i)
                for i in server.cache:
                    s_.append(i)
                s_ = np.array(s_)
                # print("next_obs_server_n_cache",next_obs_server_n_cache)
                agent_small.replay_buffer.push((s, action_small,
                                                s_, reward_new))
                obs = next_obs_small
                ep_reward_small += reward_new
                ep_time += time
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",reward_average_small)
                if (step + 1) % cache_fre == 0:
                    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    swithc_cache_value = cache_value
                    #print("swithc_cache_value", swithc_cache_value)
                    constraint_over_cache_cap = 0
                    cache_size = 0
                    for i, task in enumerate(server.cache):
                        if task == 1:
                            cache_size += env.task[i].get_cache_size
                    if cache_size > server.get_cache_cap:
                        constraint_over_cache_cap = (cache_size - server.get_cache_cap) / (1024 * 1024 * 1024 * 8)
                    #print("constraint_over_cache_cap", constraint_over_cache_cap)
                    #print("reward_average_small",reward_average_small,np.mean(reward_average_small))
                    reward_cache = np.mean(reward_average_small) - swithc_cache_value - constraint_over_cache_cap
                    next_obs_cache = obs
                    # save experience
                    agent_server_cache.replay_buffer.push((obs, action_server_cache,
                                                           next_obs_cache, reward_cache))
                    ep_reward_cache += reward_cache
                    ep_cache_cost += swithc_cache_value
                    reward_average_small = []
                    obs = next_obs_cache
                game_step += 1
                if game_step >= arglist.learning_start_step:
                    if game_step % arglist.learning_fre == 0:
                        agent_small.update(arglist)
                        # # cache开始训练的时间点： 自己设计呢
                if game_step_cache >= arglist.learning_start_step:
                    if game_step_cache % arglist.learning_fre == 0:
                        agent_server_cache.update(arglist)

        # print("step_qoe",step_qoe)
        reward_small_ra.append(ep_reward_small)
        users_time.append(ep_time)
        rewards_server_cache.append(ep_reward_cache)
        server_switch_cost.append(ep_cache_cost)

    return users_time,reward_small_ra ,rewards_server_cache,server_switch_cost


if __name__ == '__main__':
    arglist = parse_args()
    num_server = 2
    num_user = 20
    # agent1, agent2, agent_user,a1,a2,a6,a10,a17,a19,all_a = train(arglist)
    users_time, reward_small_ra, rewards_server_cache, server_switch_cost = train(arglist)
    with open("simulation/users_time_SAME.txt", "w") as f:
        for r in users_time:
            f.write(str(r) + '\n')
    with open("simulation/reward_small_SAME.txt", "w") as f:
        for cost in reward_small_ra:
            f.write(str(cost) + '\n')

    with open("simulation/rewards_server_cache_SAME.txt", "w") as f:
        for ep_time in rewards_server_cache:
            f.write(str(ep_time) + '\n')
    with open("simulation/server_switch_cost_SAME.txt", "w") as f:
        for ep_energy in server_switch_cost:
            f.write(str(ep_energy) + '\n')
    print('end')


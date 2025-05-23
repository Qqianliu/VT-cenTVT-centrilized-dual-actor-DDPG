# time 20230327
# by qian
# Use ddpg-large actor  to make decisions on cache bandwidth, computing resource allocation, and computing offloading ratio on a large time scale and actor small on a small time scale! !
# The state here is the global state, which includes both the instantaneous environment state and the long-term state; the network includes two actors, and a critic action is also two, one changes and the other does not change
# The reward function is also global.
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
from model_twin_actor import DDPG
from env import MultiAgentEnv
from my_world import Scenario
from collections import namedtuple

def make_off_env():
    # make the environment for offload & bandwidth allocation & caching
    scenario = Scenario()
    print("create the env ")
    world = scenario.make_world()
    # def __init__(self, world, reset_callback_small=None ,rest_callback_cache=None, observation_callback_small = None,observation_callback_cache = None ):
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
    # The first step is to obtain the state and action dimensions of the server agent to generate the agent:
    #print("env.observation_space_server_cache",env.observation_space_server_cache[0])
    obs_shape_server_cache = env.observation_space[0].shape[0]
    print('obs_shape_n_server_cache',obs_shape_server_cache)
    action_shape_server_cache = env.action_space_server_cache[0].shape[0]
    print('action_shape_n_server', action_shape_server_cache)
    #obs_shape_small = env.observation_space_small[0].shape[0]
    #print('obs_shape_n_server_cache', obs_shape_small)
    action_shape_small = env.action_space_small[0].shape[0]
    print('action_shape_n_server', action_shape_small)
    # a twin actor ddpg agent
    # ddpg
    twin_actor_agent = DDPG(obs_shape_server_cache,action_shape_small,action_shape_server_cache,arglist)
    print('step 2 The {} agents are inited ...'.format(1))

    print('step 3 starting iterations ...')
    cache_fre = 10
    game_step = 0
    game_step_cache = 0
    # Utility of users and servers at each step
    users_time = []
    server_switch_cost = []
    rewards_server_cache = []
    reward= []

    # At the beginning, both the upper and lower layers are reset!!! Just use reset_high
    # Get the initialized state
    # max_epsode 500
    # each_episode: 100 step, each 5 step Do caching one time, 100 times this offload and bandwidth allocation decision
    var_cache = 1
    var_small = 1
    for episode_gone in range(arglist.max_episode):
        env.world.episode = episode_gone
        obs_total = env.reset_high()
        action_cache = 0
        reward_cache = 0
        action_small = 0
        reward_small = 0

        ep_cache_cost =0.0
        ep_reward_total =0.0
        ep_reward_small =0.0
        ep_time = 0.0
        reward_average_small = []
        action_n_server_cache = 0
        for step in range(arglist.per_episode_max_len):  # The number of steps in each round is 20 * 5
            print("this is the step " + str(step) + " of episode " + str(episode_gone))
            env.world.game_step = game_step
            server = env.server
            past_cache = server.cache
            if step % cache_fre == 0:
                var_cache *= 0.9882
                var_small *= 0.9995
                # 500 step 0.988
                past_cache = server.cache
                #print("obsivation:",obs_total)
                action_server_cache = twin_actor_agent.select_action_large(obs_total)
                #print("action_server_cache",action_server_cache)
                action_n_server_cache = np.clip(np.random.normal(action_server_cache, var_cache), -1, 1)
                #print("noise: action_server_cache", action_n_server_cache)
                #env._set_action_server_cache(action_n_server_cache)
                env._set_action_ramdom_cache(action_n_server_cache)
                # offloading ,computing frequence and bandwidth 决策
                # print("obsivation:obs_small", obs_total)
                action_small = twin_actor_agent.select_action_small(obs_total)
                #print("action_small", action_small)
                action_small = np.clip(np.random.normal(action_small, var_small), -1, 1)
               # print("noise: action_small", action_small)
                env._set_action_small(action_small)
                #env._set_action_small_ave(action_small)
                #culculate the average delay of small steps
                #print("type action_small",type(action_small))
                action_total = []
                for i in action_small:
                    action_total.append(i)
                for i in action_n_server_cache:
                    action_total.append(i)
                action_total = np.array(action_total)
                print(type(action_small))

                reward_small, time = env._get_reward()
                print("the small reward :", reward_small)
                swithc_cache_value = 0
                current_cache = server.cache
                # print("current_cache", current_cache)
                # print("past_cache", past_cache)
                for i in range(len(current_cache)):
                    if current_cache[i] == 1 and past_cache[i] == 0:
                        communcation_cost = server.get_power * (env.task[i].get_cache_size / server.backhaul_rate)
                        # print("server.get_power",server.get_power,)
                        # print("env.task[i].get_cache_size / server.backhaul_rate",env.task[i].get_cache_size / server.backhaul_rate)
                        # print("communcation_cost",communcation_cost)
                        swithc_cache_value += communcation_cost/(cache_fre *20)
                #print("swithc_cache_value", swithc_cache_value)
                constraint_over_cache_cap = 0
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.task[i].get_cache_size
                if cache_size > server.get_cache_cap:
                    constraint_over_cache_cap = (cache_size - server.get_cache_cap) / (1024 * 1024 * 1024 * 8)
                #print("constraint_over_cache_cap", constraint_over_cache_cap)
                # print("reward_average_small", reward_average_small, np.mean(reward_average_small))
                reward_total = reward_small - swithc_cache_value - constraint_over_cache_cap
                reward_total1 = reward_small

                #Update the user's request size in each time slot
                env.world.step()

                next_obs = env._get_obs()

                # print("next_obs_server_n_cache",next_obs_server_n_cache)
                twin_actor_agent.replay_buffer.push((obs_total, action_total,
                                                       next_obs, reward_total))
                obs_total = next_obs

                ep_reward_total += reward_total
                ep_time += time
                ep_cache_cost += swithc_cache_value
                #obs_n_server_cache = next_obs_server_n_cache

                game_step += 1
                game_step_cache +=1
                if game_step >= arglist.learning_start_step:
                    if game_step % arglist.learning_fre == 0:
                        twin_actor_agent.update(arglist)

            else:
                print("single step !!!")
                # offloading ,computing frequence and bandwidth 决策
                #print("obsivation:obs_small", obs_total)
                action_small = twin_actor_agent.select_action_small(obs_total)
                #print("action_small", action_small)
                action_small = np.clip(np.random.normal(action_small, var_small), -1, 1)
                #print("noise: action_small", action_small)
                env._set_action_small(action_small)
                #env._set_action_small_ave(action_small)
                reward_small, time = env._get_reward()
                #print("the small reward :", reward_small)
                swithc_cache_value = 0
                current_cache = server.cache
                #print("current_cache", current_cache)
                #print("past_cache", past_cache)
                for i in range(len(current_cache)):
                    if current_cache[i] == 1 and past_cache[i] == 0:
                        communcation_cost = server.get_power * (env.task[i].get_cache_size / server.backhaul_rate)
                        swithc_cache_value += communcation_cost / (cache_fre * 20)
                #print("swithc_cache_value", swithc_cache_value)
                constraint_over_cache_cap = 0
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.task[i].get_cache_size
                if cache_size > server.get_cache_cap:
                    constraint_over_cache_cap = (cache_size - server.get_cache_cap) / (1024 * 1024 * 1024 * 8)
                #print("constraint_over_cache_cap", constraint_over_cache_cap)

                reward_total = reward_small - swithc_cache_value - constraint_over_cache_cap
                reward_total1 = reward_small
                env.world.step()
                next_obs = env._get_obs()
                action_total = []
                for i in action_small:
                    action_total.append(i)
                for i in action_n_server_cache:
                    action_total.append(i)
                action_total = np.array(action_total)
                twin_actor_agent.replay_buffer.push((obs_total, action_total,
                                                     next_obs, reward_total))
                obs_total = next_obs
                ep_reward_total += reward_total
                ep_time += time
                ep_cache_cost += swithc_cache_value
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",ep_reward_total)
                game_step += 1
                if game_step >= arglist.learning_start_step:
                    if game_step % arglist.learning_fre == 0:
                        twin_actor_agent.update(arglist)
                    

        # print("step_qoe",step_qoe)
        reward.append(ep_reward_total)
        users_time.append(ep_time)
        server_switch_cost.append(ep_cache_cost)

    return users_time,reward,server_switch_cost


if __name__ == '__main__':
    arglist = parse_args()
    # agent1, agent2, agent_user,a1,a2,a6,a10,a17,a19,all_a = train(arglist)
    users_time, reward,server_switch_cost= train(arglist)
    with open("simulation1/users_time.txt", "w") as f:
        for r in users_time:
            f.write(str(r) + '\n')
    with open("simulation1/reward.txt", "w") as f:
        for cost in reward:
            f.write(str(cost) + '\n')
    with open("simulation1/server_switch_cost", "w") as f:
        for ep_energy in server_switch_cost:
            f.write(str(ep_energy) + '\n')
    print('end')


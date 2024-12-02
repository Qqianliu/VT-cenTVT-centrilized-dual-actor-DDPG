# 20230324/ 20230409
# by qian
# 这个就是搭建完自己的环境之后。然后创建一个多智能体学习的更新的环境框架,

import gym
from gym import spaces
import numpy as np
import cvxpy as cvx


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback_small=None ,rest_callback_cache=None,observation_callback = None, reward = None ):

        self.world = world
        # print(self.agents)
        ## 这里是列表
        self.user = self.world.User
        self.server =  self.world.Server[0]
        self.task = self.world.Tasks

        self.n_user = len(world.User)
        self.n_server = len(world.Server)
        self.n_task = world.num_task
        # 奖励函数和重置函数以及观测函数
        self.rest_callback_small = reset_callback_small
        self.reset_callback_server_cache = rest_callback_cache
        #self.observation_callback_small = observation_callback_small
        #self.observation_callback_server_cache = observation_callback_cache
        self.observation_callback = observation_callback
        self.reward_callback = reward
        self.time = 0
        # configure spaces for agent server
        self.action_space_server_cache = []
        #self.observation_space_server_cache = []
        self.action_space_small = []
        #self.observation_space_small = []
        self.observation_space = []

        # 定义缓存的动作和状态空间
        ac_dim_caching  = world.num_task
        self.action_space_server_cache.append(spaces.Box(low = -1, high = 1,shape=(ac_dim_caching,),
                                        dtype=np.float32))
        obs_dim = len(self.observation_callback(self.world))
        # question: why not define obs_dim directly
        #self.observation_space_server_cache.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        self.observation_space.append(
            spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        # action： bandwidth, computing,  offloading proportion
        ac_dim_small = world.num_User*3
        self.action_space_small.append(spaces.Box(low =-1 , high = 1,shape=(ac_dim_small,),
                                        dtype=np.float32))
        obs_dim = len(self.observation_callback(self.world))
        # question: why not define obs_dim directly
        #self.observation_space_small.append(
            #spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
    # according to the action, change the environment , acquire the new observation and reward








    # 这个是大尺度的重置的，就是相关缓存
    def reset_high(self):
        print("reset total environment !!!")
        # reset 全部的环境，就是先重置一下AP 的位置，然后把AP 的缓存置空，更新用户的位置，请求，大小，
        # 主要是改变world 中的三个变量
        self.reset_callback_server_cache(self.world)
        self.rest_callback_small(self.world)
        # record observations for each agent
        #obs_server_cache = self._get_obs_server_cache()
        #obs_small = self._get_obs_small()
        #obs_n_server_cache.append(self._get_obs_server_cache())
        obs = self._get_obs()
        return obs

    def _set_action_ramdom_cache(self,action):
        random_action = np.random.random(len(action))
        cache = []
        for i in action:
            if i >= 0.5:
                cache.append(1)
            else:
                cache.append(0)
        server = self.world.Server[0]
        server.cache = cache
        print("after the maping :server 1 .cache", server.cache)





    def _set_action_server_cache(self, action):
        #print("action",action)
        cache = []
        for i in action:
            if i >= 0:
                cache.append(1)
            else:
                cache.append(0)
        server = self.world.Server[0]
        server.cache = cache
        print("after the maping :server 1 .cache",server.cache)

    def _set_action_small_ave(self, action):
        #print("the original action :", action)
        for i in range(len(action)):
            action[i] = 0.5
        #print("the changed action",action)
        n = self.world.num_User
        bandwidth  = action[0:n]
        print("bandwidth",bandwidth)
        offload_proportion = action[n: 2*n]
        print("offload_proportion",offload_proportion)
        computing_fre = action[2*n: 3*n]
        print("computing_fre", computing_fre)

        # set bandwith for ES and each user
        # regular the bandwidth allocation factor:
        # check if existes 0: change to 0.01
        for i in range(len(bandwidth)):
            if bandwidth[i] == 0:
                bandwidth[i] = 0.1
        sum_bandwidth = sum(bandwidth)
        for i in range(len(bandwidth)):
            bandwidth[i] = bandwidth[i]/sum_bandwidth

        server = self.world.Server[0]
        server.bandwidth_allocation = bandwidth
        print("after maping bandwidth_allocation", server.bandwidth_allocation)
        for i,user in enumerate( self.world.User):
            user.bandwidth = bandwidth[i]
        cache = server.get_cache
        # print("cache",cache)
        # set computing_fre and offload propotion : need considering service caching
        for i in range(len(offload_proportion)):
            user = self.world.User[i]
            #print("cache:",cache)
            #print("user.get_request",user.get_request)
            if cache[user.get_request-1] ==0:
                offload_proportion[i] = 0
        for i in range(len(offload_proportion)):
            if offload_proportion[i] == 0:
                computing_fre[i] = 0
            elif computing_fre[i] ==0:
                #
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!change the wrong action",computing_fre)
                computing_fre[i] = 0.1


        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^sum_comp_fre",computing_fre)
        sum_comp_fre = sum(computing_fre)
        if sum_comp_fre ==0:
            pass
        else:
            for i in range(len(computing_fre)):
                computing_fre[i] = computing_fre[i]/sum_comp_fre

        server.offload_proportion = offload_proportion
        server.computing_resource = computing_fre
        print("after maping offload_proportion", server.offload_proportion)
        print("after maping computing_resource", server.computing_resource)
        for i,user in enumerate(self.world.User):
            user.comp = computing_fre[i]
            user.offload = offload_proportion[i]

    def _set_action_small(self, action):
        #print("the original action :", action)
        for i in range(len(action)):
            action[i] = abs(action[i])
        #print("the changed action",action)
        n = self.world.num_User
        bandwidth  = action[0:n]
        print("bandwidth",bandwidth)
        offload_proportion = action[n: 2*n]
        print("offload_proportion",offload_proportion)
        computing_fre = action[2*n: 3*n]
        print("computing_fre", computing_fre)

        # set bandwith for ES and each user
        # regular the bandwidth allocation factor:
        # check if existes 0: change to 0.01
        for i in range(len(bandwidth)):
            if bandwidth[i] == 0:
                bandwidth[i] = 0.1
        sum_bandwidth = sum(bandwidth)
        for i in range(len(bandwidth)):
            bandwidth[i] = bandwidth[i]/sum_bandwidth

        server = self.world.Server[0]
        server.bandwidth_allocation = bandwidth
        print("after maping bandwidth_allocation", server.bandwidth_allocation)
        for i,user in enumerate( self.world.User):
            user.bandwidth = bandwidth[i]
        cache = server.get_cache
        # print("cache",cache)
        # set computing_fre and offload propotion : need considering service caching
        for i in range(len(offload_proportion)):
            user = self.world.User[i]
            #print("cache:",cache)
            #print("user.get_request",user.get_request)
            if cache[user.get_request-1] ==0:
                offload_proportion[i] = 0
        for i in range(len(offload_proportion)):
            if offload_proportion[i] == 0:
                computing_fre[i] = 0
            elif computing_fre[i] ==0:
                #
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!change the wrong action",computing_fre)
                computing_fre[i] = 0.1


        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^sum_comp_fre",computing_fre)
        sum_comp_fre = sum(computing_fre)
        if sum_comp_fre ==0:
            pass
        else:
            for i in range(len(computing_fre)):
                computing_fre[i] = computing_fre[i]/sum_comp_fre

        server.offload_proportion = offload_proportion
        server.computing_resource = computing_fre
        print("after maping offload_proportion", server.offload_proportion)
        print("after maping computing_resource", server.computing_resource)
        for i,user in enumerate(self.world.User):
            user.comp = computing_fre[i]
            user.offload = offload_proportion[i]


    def _get_obs(self):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(self.world)

    # def _get_obs_small(self):
    #     if self.observation_callback_small is None:
    #         return np.zeros(0)
    #     return self.observation_callback_small(self.world)
    #
    #
    # def _get_obs_server_cache(self):
    #     if self.observation_callback_server_cache is None:
    #         return np.zeros(0)
    #     return self.observation_callback_server_cache(self.world)
    def _get_reward(self):
        if self.reward_callback is None:
            return np.zeros(0)
        return self.reward_callback(self.world)


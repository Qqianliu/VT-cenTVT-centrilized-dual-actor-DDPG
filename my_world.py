# 相关自己场景的一些具体的参数设置即构建一个自己的相关的环境，以及重置环境，获取归一化状态，得到奖励

import numpy as np
from core import  *
from scenario import BaseScenario

class Scenario(BaseScenario):
        def make_world(self):
            print("create the entities in the world")
            world = World()
            world.cache_decision_fre = 2
            # how many steps, update the  task which user request
            world.task_update_fre = 1
            world.white_noise = -114  # dbm
            # list of agents and entities (can change at execution-time!)
            world.num_User = 10
            world.num_Server = 1
            world.num_task = 10
            self.User = []
            self.Server = []
            self.Tasks = []
         # add
            # edge server id from 1: location :two dimension ; com_cap: 20GHZ; pow : 1w
            # bandwidth_cap: 5MHZ; cache and next cache = [0];
            # cache_cap :300GB ;backhual_rate : 100Mbps（100mpbs 云端很远；）;
            # 在初始化定义时，所有的单位都是基本单位。bits\
            # def __init__(self, index, type, loc, power, com_cap, bandwidth_cap, cache, next_cache, cache_cap,
             #            backhaul_rate):
            world.Server = [Server((i + 1), 'server%d' % (i + 1), [125,125],1, 20 * (10 ** 9) ,
                                   20 * (10 ** 6), np.zeros(world.num_task), np.zeros(world.num_task),
                                              200 * (1024 * 1024 * 1024 * 8), 200 * (1024 * 1024)) for i in
                                 range( world.num_Server)]
            # print("!!!!!!!!!!!!!!!!!!!!!!!",world.Server[1].cache)
            # def __init__(self, index, type, loc,  power, compfre):
            # user id from 0; location :fixed two dimension; power:20 dbm ; comp 1*10^9
            world.User = [User(i, 'user %d' % i, np.zeros(2), 20,1 * (10 ** 9)) for i in range( world.num_User)]

            # def __init__(self, index, cache_size 20MB-400MB , com_fre 400cycle/1000cycle, size, 50kB- 500KB)
            world.Tasks = [Task((i + 1), 20, 400, 50 )for i in range(world.num_task)]

            with open("task_characteristic.txt", "w") as f:
                total_size = 0
                for task in world.Tasks:
                    total_size+=task.get_cache_size
                    f.write(str(task.get_id) + ' ' + str(task.get_cache_size) + ' ' + str(task.size)
                            +' '+str(task.get_cycle)+'' +str(task.get_cache_gain_value)+'\n')
                f.write("total_size"+str(total_size)+"\n" +str(total_size/(1024*1024*1024*8)))
            # initialize the location of user and server
            # edge server is 5G BS ,the coverage diameter is 250 m
            #  so the network is 500 * 750
            # self.reset_world(world)
            self.reset_cache(world)
            self.reset_world(world)
            print("make world end")
            return world


        def reset_world(self, world):
            episode = world.episode
            print("reset !!!, the is episode ", episode)
            np.random.seed(episode)
            # AP范围35 - 300中。
            # 更新用户的位置
            for i in range(world.num_User):
                loc = np.random.uniform(low=0, high=250, size=2)
                world.User[i].loc = loc
            # 重置用户的请求
            for user in world.User:
                user.request = np.random.randint(1, 11)
                user.task_size = world.Tasks[user.request - 1].get_size
                user.task_cycle = world.Tasks[user.request - 1].get_cycle
            # 动作影响的状态：在最初的时候，我觉得 是可以重置为0 的；
            # #计算一下分配给每种缓存任务的计算频率，也暗含了缓存决策
            task_com_fre = np.zeros(world.num_task)
            # 计算一下分配给每种任务的带宽，
            task_band = np.zeros(world.num_User)
            # 计算一下每种任务的卸载比例，就暗含了缓存决策；
            task_offload_step = np.zeros(world.num_task)
            task_cache_gain_step = np.zeros(world.num_task)
            # 计算一下,更新用户的请求之后，环境的随机量：
            # 用户的单步请求
            task_request = []
            # 用户的请求的统计的缓存增益
            task_cache_gain_step = np.zeros(world.num_task)
            for user in world.User:
                task_request.append(user.get_request)

            world.task_com_fre = task_com_fre
            world.task_band = task_band
            world.task_input_size = task_request

            world.task_caching_value_step = task_cache_gain_step
            world.task_offload_step = task_offload_step
            world.task_caching_value = world.task_caching_value_step
            world.task_offload = world.task_offload_step
            print("rest world end ")

        #  这里重置一下AP的位置，以及AP 的缓存制空，开启新一轮的重置
        def reset_cache(self,world):
            world.Server[0].loc = [125, 125]
            cache = np.zeros(world.num_task)
            for server in world.Server:
                server.cache = cache
                server.next_cache = cache

        def reward(self,world):
            time = 0
            task_cache_gain_step = np.zeros(world.num_task)
            for user in world.User:
               t , t_cloud  = self.get_task_delay_cloud(user, world)
               cache_gain = t_cloud - t
               print(cache_gain)
               task_cache_gain_step[user.get_request-1] += cache_gain
               # 在更新奖励函数这里更新任务的缓存价值
               print("t",t)
               time +=t
            if world.task_caching_value is None:
                world.task_caching_value = task_cache_gain_step
            else:
                for i in range(len(world.task_caching_value)):
                    world.task_caching_value[i] = (world.task_caching_value[i] + task_cache_gain_step[i])/2
            reward = - time/ world.num_User
            return reward, time

        # 反馈奖励 在给定当前缓存状态条件下， 用户和服务器都完成动作的分配之后，就进行奖励函数的计算

        def get_task_delay_cloud(self, user, world):
            task_size = user.get_task_size
            task_cycle = user.get_task_cycle
            server = world.Server[0]
            b = user.bandwidth * server.get_bandwidth_cap
            # 因为用户的位置固定，这里只计算了大尺度的衰落
            channel_gain = world.get_gain(server, user)
            # print("the channel id and the channel gain", channel_id,channel_gain)
            power = pow(10, (user.get_power - 30) / 10)  # 20dbm to w - 0.1w
            self_gain = channel_gain * power
            # print("self_gain multiplex power", self_gain)
            white_noise_pow = pow(10, (world.white_noise - 30) / 10)  # -144 dbm to w
            t_trans_e = task_size / (b * np.log2(1 + self_gain / white_noise_pow))
            t_trans_c = task_size/(server.backhaul_rate/world.num_User)
            #print("rate: wireless",(b * np.log2(1 + self_gain / white_noise_pow)) )
            #print("wired link ", (server.backhaul_rate/world.num_User))
            t_cloud = t_trans_e + t_trans_c + task_cycle / (4 * 10 ** 9)
            print("t_cloud",t_cloud)
            #print("t_trans_e",t_trans_e)
            if server.cache[user.get_request - 1] == 1:
                # print("user.get_comp",user.get_comp)
                com_fre = server.get_com_cap * user.get_comp
                # print("com_fre",server.get_com_cap)
                t_e = task_cycle * user.get_offload / com_fre
                t_c =  (task_size * (1 - user.get_offload) / (server.backhaul_rate/world.num_User)) + task_cycle * (1 - user.get_offload) / (4 * 10 ** 9)
                t_ = [t_e,t_c]
                t_exe = max(t_)
                #print("vedge, cloud,",t_)
                #print("t_exe",t_exe)
                t_total = t_exe + t_trans_e
                # print("get_offload",user.get_offload)
                # print("get_comppre",user.get_comp)
                print("edge computing ", t_total)
            else:
                t_total = t_cloud
                print("cloud computing ", t_total)
            return t_total, t_cloud




        # 这里是获得状态，并且进行归一化的部分， 是相关我们自己的环境进行的！！！
        def observation_small(self, world):
            o = []
            for i in world.task_com_fre:
                o.append(i)
            for i in world.task_band:
                o.append(i)
            for i in world.task_input_size:
                o.append(i/10)
            obs = np.array(o)
            return obs

        def observation_large(self,world):
            o = []
            print("world.task_caching_value",world.task_caching_value)
            for i in world.task_caching_value:
                o.append(i)
            for i in world.task_offload:
                o.append(i)
            obs = np.array(o)
            return obs

        def observation_total(self, world):
            o = []
            # for i in world.task_com_fre:
            #     o.append(i)
            # for i in world.task_band:
            #     o.append(i)
            # for i in world.task_input_size:
            #     o.append(i / 10)

            for i in world.task_caching_value:
                o.append(i)
            for i in world.task_offload:
                o.append(i)
            obs = np.array(o)
            min_obs = min(obs)
            max_obs = max(obs)
            if min_obs == max_obs:
                pass
            else:
                for i in range(len(obs)):
                    obs[i] = (max_obs - obs[i])/(max_obs-min_obs)
            return obs




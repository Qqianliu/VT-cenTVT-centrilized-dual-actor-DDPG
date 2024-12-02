# joint 3c resource allocation in cloud - edge - end env
# BY qian
# time 20230323/ 20230409
# 希望是一个比较顺利的程序编写完成一篇小论文，然后顺顺利利的去开会，毕业
# 希望可以在这个5天之内把这个程序调好一些，然后在修改好，顺利投出去，
# by lq
import numpy as np
from collections import Counter


# state of small timesscale resource allocation
class State_small(object):
    def __init__(self):
        # state
        # {task_com_fre = each type task allocated computing frequency lost time slot,  input-size ,cache_state of each server}
        #  state dim = service_num * 2 + user_num
        self.task_com_fre = None
        self.task_band = None
        self.task_input_size = None



class Action_small(object):
    def __init__(self):
        # action of agent offloading decision propotion of tasks processing at ES
        # action_dim = 3* number_user
        self.offload = None
        self.band = None
        self.comp = None


# large timescale level (caching )
class State_cache(object):
    def __init__(self):
        # state
        # {user_offloading task_caching gain value }
        # state_dim :  2 * num_task
        self.task_caching_value= None
        self.task_offload = None


class Action_cache(object):
    def __init__(self):
        # action of caching  )
        # action dim : num_service
        self.cache = None

# properties of agent server small ( offlaoding proportion bandwidth ,computing )
class Agent_Server_small(object):
    def __init__(self):
        self.ac_range = 1
        self.state_small = State_small()
        self.action_small = Action_small()


# properties of agent Server cache (action : cache)
class Agent_Server_large(object):
    def __init__(self):
        self.ac_range = 1
        self.state_large = State_cache()
        self.action_large = State_cache()


# properties and state of the physical world entity
class Entity(object):
    def __init__(self, index, type, loc, power):
        self.index = index
        self.type = type
        self.loc = loc
        self.power = power


    @property
    def get_id(self):
        return self.index

    @property
    def get_type(self):
        return self.type

    @property
    def get_loc(self):
        return self.loc
    @property
    def get_power(self):
        return self.power


# properties of agent User (request contents)
class User(Entity):
    def __init__(self, index, type, loc,  power, compfre):
        Entity.__init__(self, index, type, loc, power)
        # request which content (size, cycle)
        self.time_th = 0.2
        self.request = None
        self.task_size = None
        self.task_cycle = None
        self.compfre = compfre
        # action
        self.bandwidth  = 0
        # offloading to (proportion)
        self.offload = None
        self.comp = None

        self.uplink_rate = None

        # script behavior to execute
        self.action_callback = None
    @property
    def get_compfre(self):
        return self.compfre
    @property
    def get_task_size(self):
        return self.task_size
    @property
    def get_task_cycle(self):
        return self.task_cycle
    # request which content (size, cycle)
    @property
    def get_request(self):
        return self.request
    @property
    def get_bandwidth(self):
        return self.bandwidth
    @property
    def get_comp(self):
        return self.comp
    @property
    def get_offload(self):
        return self.offload
    @property
    def get_time_th(self):
        return self.time_th


# properties of anget server com ( action bandwidth allocation  and caching )
class Server(Entity):
    def __init__(self, index, type, loc, power, com_cap, bandwidth_cap, cache, next_cache, cache_cap, backhaul_rate):
        Entity.__init__(self, index, type, loc, power)
        self.com_cap = com_cap  # GHz 转 Hz
        self.bandwidth_cap =  bandwidth_cap                       # 20 * (10 ** 6)  # 带宽 20MHz 转 Hz
        self.cache_cap = cache_cap
        self.backhaul_rate = backhaul_rate
        self.ac_range = 1
        # ES action
        self.cache = cache  # 当前时隙，本身缓存的东西
        self.next_cache = next_cache  # 当前时隙，动作做出的缓存
        self.computing_resource = None
        self.offload_proportion = None
        self.bandwidth_allocation = None
        #  small states
        self.task_com_fre = None
        self.task_input_size = None
        self.task_band = None
        # large state for caching
        self.task_caching_value = None
        self.task_offload  = None

    # get the  allocation actions
    @property
    def get_cache(self):
        return self.cache
    @property
    def get_bandwiddth_allocation(self):
        return self.bandwidth_allocation
    @property
    def get_computing_resource(self):
        return self.computing_resource
    @property
    def get_offload_proportion(self):
        return self.offload_proportion
    @property
    def get_bandwidth_cap(self):
        return self.bandwidth_cap
    @property
    def get_cache_cap(self):
        return self.cache_cap
    @property
    def get_backhual_rate(self):
        return self.backhaul_rate
    @property
    def get_com_cap(self):
        return self.com_cap

    @property
    def get_next_cache(self):
        return self.next_cache




class Task(object):
    def __init__(self, index,cache_size,com_fre,size):
        # task  都是固定的属性，就是按照index 的生计算和通信的复杂度以及缓存的复杂度，所以都是通过一个初始的值开始进行叠加的
        self.index = index
        self.cache_size = cache_size*index * 2 * 1024 * 1024 *8  # GB 表示数据容量 =1024*1024 *1024
        self.randomsize = cache_size*index
        self.com_fre = com_fre + (index*50)
        self.size = size * index * 1024 * 8  # kB 1024*8
        self.cycle = self.com_fre *self.size
        # 1000 = computing_max / cache_low
        self.cache_gain_value = ( self.cycle /self.cache_size)

    @property
    def get_id(self):
        return self.index
    @property
    def get_size(self):
        return self.size
    @property
    def get_cycle(self):
        return self.cycle
    @property
    def get_cache_size(self):
        return self.cache_size # GB  G表示数据容量 =1024*1024*1024*8  表示传输速率 10**9
    @property
    def get_cache_gain_value(self):
        return self.cache_gain_value


# multi-agent world
class World(object):
    def __init__(self):
        self.episode = 0
        self.episode_cache = 0
        self.game_step_cache = 0
        self.game_step = 0
        # how many steps, do one caching decision
        self.cache_decision_fre  =0
        # how many steps, update the  task which user request
        self.task_update_fre = 0
        self.white_noise = 0  # dbm
        # list of agents and entities (can change at execution-time!)
        self.num_User = 0
        self.num_Server =0
        self.num_task  = 0
        self.alpha = 0
        self.beta = 0
        self.xi = 0
        self.User = []
        #  this only one server
        self.Server = []
        self.Tasks = []
        # position dimensionality
        self.dim_p = 2
        self.tStoC = 0.02  # 100-200ms
        #  small states
        self.task_com_fre = None
        self.task_input_size = None
        self.task_band = None
        # large state for caching
        self.task_caching_value_step = None
        self.task_offload_step = None
        self.task_caching_value = None
        self.task_offload = None

    # update state of the world，单步环境更新
    def step(self):
        # 单步更新，更新用户请求任务的类型，更新动作，更新状态
        # 在用户执行完 卸载决策，带宽，计算资源分配，就是更新整个环境的参量


        # 动作影响的状态：

        # 计算一下分配给每种个用户的带宽，
        task_band = np.zeros(self.num_User)
        # #计算一下分配给每种缓存任务的计算频率，也暗含了缓存决策
        task_com_fre = np.zeros(self.num_task)
        # 计算一下每种任务的卸载比例，就暗含了缓存决策；
        task_offload_step = np.zeros(self.num_task)
        for i,user in enumerate( self.User):
            # request id is start from 1 -10
            user_request = user.get_request -1
            task_com_fre[user_request] += user.get_comp
            task_band[i] += user.get_bandwidth
            task_offload_step[user_request] += user.get_offload
        # 更新用户的请求
        for user in self.User:
            user.request = np.random.randint(1, 11)
            user.task_size = self.Tasks[user.request-1].get_size
            user.task_cycle = self.Tasks[user.request-1].get_cycle


        # 计算一下,更新用户的请求之后，环境的随机量：
        #用户的单步请求
        task_request =[]
        # 用户的请求的统计的缓存增益
        #task_cache_gain_step = np.zeros(self.num_task)
        for user in self.User:
            task_request.append(user.get_request)
            #task_cache_gain_step[user.get_request-1] += self.Tasks[user.get_request-1].cache_gain_value

        self.task_com_fre = task_com_fre
        self.task_band = task_band
        self.task_input_size = task_request

        #self.task_caching_value_step = task_cache_gain_step
        self.task_offload_step = task_offload_step
        if self.task_caching_value is None:
          #self.task_caching_value = self.task_caching_value_step
          self.task_offload = self.task_offload_step
        else:
            for i in range(len(self.task_caching_value)):
                #self.task_caching_value[i] = (self.task_caching_value[i] + self.task_caching_value_step[i])/2
                self.task_offload[i] = (self.task_offload[i] + self.task_offload_step[i])/2




    def get_gain(self, server, user):
        ch_gains = 0
        server_loc = server.get_loc
        user_loc = user.get_loc
        # print('获取MBS与对应用户的增益', bs_loc, user_loc, bs_loc[0], user_loc[0], bs_loc[2], user_loc[1])
        distance = pow(pow((server_loc[0] - user_loc[0]), 2) + pow((server_loc[1] - user_loc[1]), 2), 0.5)
        ch_gains = 128.1 + 37.6 * np.log10(distance / 1000)  # m 转 distance KM
        # print("ch_gains:***************************ch_gains:")
        # print(ch_gains)
        # shadow = np.random.lognormal(0, 8, 1)
        # print(ch_gains, shadow)
        # shadow = 0
        ch_gains = pow(10, -ch_gains / 10)  # db  to w
        rayleigh = np.random.rayleigh(1)
        #rayleigh = 1
        gains = ch_gains * rayleigh  # w
        # print("gain",gains)
        return gains
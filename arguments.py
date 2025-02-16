import time
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.is_available())
time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments")
    # environment
    parser.add_argument("--scenario_name", type=str, default="3c_environment", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--per_episode_max_len", type=int, default= 100, help="two layers maximum episode length")
    parser.add_argument("--per_episode",type=int,default=20, help=" each episode has max steps ")
    parser.add_argument("--max_episode", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")  # 对手人数
    ## ppo 算法
    parser.add_argument("--clip_param", type=float, default=0.2, help = "epislon : the tolerable difference "
                                                                        "between the old and new policy ")
    parser.add_argument("--ppo_update_step", type=int, default=5,  help="learn times of each step")
    parser.add_argument("--buffer_capacity_ppo", type=int, default=5000, help="1 ")

    ##

    # core training parameters
    parser.add_argument("--device", default=device, help="torch device")
    parser.add_argument("--learning_start_step", type=int, default= 2000, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default = 100, help="learning frequency")
    parser.add_argument("--update_iteration", type=int, default=30, help="learn times of each step")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=0.0001, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=0.0002, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.90, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=500000, help="number of data stored in the memory")
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--num_units_openai", type=int, default=64, help="number of units in the mlp")
    parser.add_argument('--tau', default=0.005, type=float)

    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=200, help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=200, help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="save_models",
            help="directory in which training state and model should be saved")
    parser.add_argument("--old_model_name", type=str, default="models//", \
            help="directory in which training state and model are loaded")

    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", \
            help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", \
            help="directory where plot data is saved")
    return parser.parse_args()

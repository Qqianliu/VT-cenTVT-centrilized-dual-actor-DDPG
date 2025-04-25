This repository provide the code for paper Q. Liu, H. Zhang, X. Zhang and D. Yuan, "Improved DDPG Based Two-Timescale Multi-Dimensional Resource Allocation for Multi-Access Edge Computing Networks," in IEEE Transactions on Vehicular Technology, vol. 73, no. 6, pp. 9153-9158, June 2024,  
  doi: 10.1109/TVT.2024.3360943.
  keywords: {Resource management;Computational modeling;Costs;Delays;Optimization;Bandwidth;Edge computing;Service caching;multi-dimensional 
             resources allocation;multi-access edge computing;two-timescale;centralized dual-actor deep deterministic policy gradient},
This code how the proposed DRL-agent does the bandwidth, power and cache decision for a two-timescale edge computing scenario.

If you want to reproduce  our proposed centrilized-dual-actor-DDPG,  
**First, you should have a python venv,python3,torch and so on**

**Then, you  go the mian_two_scale_twin.py to reproduce our training, and save the reward.text ****
**Finally, you can use the plot_reward.py to show the results.


If you want design your own DRL-agent for different use case,
 ** you can change the model : for different DRL algorithms (model.py) **,like DDPG,DQN,PPO, in my work i chose the the orginal DDPG (model.py) and our proposed Improved-DDPG (model-twin-actor.py, and model-twin-actor1.py), here if you change the DRL algorithm, you also need to change the  (Replay_buffer.py ) for your own memory storage {s,a,s',r}.

 ** also you can change the simulation envrionment ( env.py) for other use case, doing different decisions**, in my work, we set a  envrionment( env.py) ,it has the one edge servr and some user decives which be defined in (core.py) 
 **   finally, you can design your own traning algorithm for your own DRL-agent ( main_two_scale_one/twin/twinrc.py ) 
**

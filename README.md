# Deep reinforcemnt learning for resource allocation!

# TVT-centrilized-dual-actor-DDPG
# Paper:  Q. Liu, H. Zhang, X. Zhang and D. Yuan, "Improved DDPG Based Two-Timescale Multi-Dimensional Resource Allocation for Multi-Access Edge Computing Networks," in IEEE Transactions on Vehicular Technology, vol. 73, no. 6, pp. 9153-9158, June 2024, 
# doi: 10.1109/TVT.2024.3360943.
keywords: {Resource management;Computational modeling;Costs;Delays;Optimization;Bandwidth;Edge computing;Service caching;multi-dimensional resources allocation;multi-access edge computing;two-timescale;centralized dual-actor deep deterministic policy gradient},
# Code structure
# model : the orginal DDPG (model.py) and we proposed Improved-DDPG (model-twin-actor.py, and model-twin-actor1.py)
# envrionment: env.py ,it has the one edge servr and some user decives.....
# core.py : for entity class defination
# Replay_buffer.py for memory the {s,a,s',r}
# main_two_scale_one/twin/twinrc.py for traning, sicne in the paper we have some baselines, so put all here.



import numpy as np
import imp
import os.path as osp




# defines scenario upon which the world is built  定义构建世界的场景

class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def load(self,name):
        pathname = osp.join(osp.dirname(__file__), name)
        return imp.load_source('', pathname)


from collections import OrderedDict
import matplotlib.pyplot as plt

class Logger(object):
    '''
    logs train/val results, such as loss or accuracy at each iteration
    '''
    def __init__(self, keys:list, desc:str=None):
        assert len(keys) >= 1, "must have at least one key to save"
        for key in keys: assert isinstance(key, str), "each key must be a string"
        self.keys = keys
        self.desc = desc
        self.iteration = 0
        self.data = OrderedDict((key, {'x':[], 'y':[]}) for key in self.keys)

    def increment_iteration(self, inc:int=1):
        '''
        increment the iteration by INC
        '''
        assert inc >= 1 and isinstance(inc, int)
        self.iteration += inc

    def record(self, key:str, val:float):
        '''
        record {KEY:VAL} at the current iteration
        '''
        assert key in self.keys
        self.data[key]['x'].append(self.iteration)
        self.data[key]['y'].append(val)

    def plot(self, keys:list=None, logy=False):
        '''
        plot those keys specified
        '''
        plot = plt.semilogy if logy else plt.plot
        if not keys: keys = self.keys
        for key in keys:
            plot(self.data[key]['x'], self.data[key]['y'])
        plt.legend(keys)
        plt.show()
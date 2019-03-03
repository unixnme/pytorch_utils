import matplotlib.pyplot as plt
import numpy as np

class Logger(object):
    '''
    logs train/val results, such as loss or accuracy at each iteration
    '''
    def __init__(self, keys:list, desc:str=None):
        assert len(keys) >= 1, "must have at least one key to save"
        for key in keys: assert isinstance(key, str), "each key must be a string"
        for key in keys: assert len(key.split()) == 1 and key.split()[0] == key
        assert len(np.unique(keys)) == len(keys), "each key must be unique"
        self.keys = keys
        self.desc = desc
        self.dt = self.dtype()
        self.iteration = 0
        self.new_row = np.ones((1,), dtype=self.dt)
        for key in self.keys: self.new_row[key] = np.nan
        self.table = self.new_row.copy()
        self.table[0]['iter'] = self.iteration

    def increment_iteration(self, inc:int=1):
        '''
        increment the iteration by INC
        '''
        assert inc >= 1 and isinstance(inc, int)
        self.iteration += inc
        if not self._last_row_empty():
            self.table = np.concatenate([self.table, self.new_row])
        self.table[-1]['iter'] = self.iteration

    def _last_row_empty(self):
        for key in self.keys:
            if self.table[-1][key] == self.table[-1][key]:
                return False
        return True

    def record(self, key:str, val:float):
        '''
        record {KEY:VAL} at the current iteration
        '''
        assert key in self.keys
        assert isinstance(val, (int, float)) and val == val

        self.table[-1][key] = val

    def plot(self, keys:list=None, logy=False):
        '''
        plot those keys specified
        '''
        plot = plt.semilogy if logy else plt.plot
        if not keys: keys = self.keys
        for key in keys:
            y = self.table[key]
            idx = ~np.isnan(y)
            x = self.table['iter'][idx]
            y = y[idx]
            plot(x,y)
        plt.legend(keys)
        plt.show()

    def save(self, filename:str):
        '''
        save the logger result as a text file
        '''
        text = "%d" % self.iteration
        for key in self.keys: text += ' %s' % key
        for row in self.table:
            text += "\n%d" % row[0]
            for key in self.keys: text += ' %f' % row[key]
        with open(filename, 'w') as f:
            f.write(text)

    @staticmethod
    def load(filename:str) -> 'Logger':
        with open(filename, 'r') as f:
            lines = f.read().split('\n')
        keys = lines[0].split()
        logger = Logger(keys[1:])
        logger.iteration = int(keys[0])
        logger.table = np.empty((len(lines)-1,), dtype=logger.dt)
        for i,line in enumerate(lines[1:]):
            values = line.split()
            logger.table[i]['iter'] = int(values[0])
            for j,key in enumerate(keys[1:]):
                logger.table[i][key] = np.float64(values[j+1])

        return logger

    def dtype(self) -> np.dtype:
        '''
        return numpy datatype suitable for the keys
        '''
        types = [('iter', np.int64)]
        for key in self.keys:
            types.append((key, np.float64))
        return np.dtype(types)

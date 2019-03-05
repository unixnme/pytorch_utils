import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

class Logger(object):
    '''
    logs train/val results, such as loss or accuracy at each iteration
    '''
    def __init__(self, keys:list, filename:str='result.log', desc:str=None):
        assert len(keys) >= 1, "must have at least one key to save"
        for key in keys: assert isinstance(key, str), "each key must be a string"
        for key in keys: assert len(key.split()) == 1 and key.split()[0] == key
        assert len(np.unique(keys)) == len(keys), "each key must be unique"
        assert not osp.exists(filename), '%s already exists'
        self.file = open(filename, 'w', buffering=1)
        self.keys = keys
        self._save_keys()
        self.desc = desc
        self.dt = self.dtype(self.keys)
        self.iteration = 0
        self.filename = filename
        self.empty_row = np.ones((1,), dtype=self.dt)
        for key in self.keys: self.empty_row[key] = np.nan
        self.table = self.empty_row.copy()
        self.table[0]['iter'] = self.iteration

    def __del__(self):
        if not self._last_row_empty():
            self._save_last_row()
        self.file.close()

    def _save_keys(self):
        for key in self.keys:
            self.file.write(key + ' ')

    def increment_iteration(self, inc:int=1):
        '''
        increment the iteration by INC
        write out the current result
        '''
        assert inc >= 1 and isinstance(inc, int)
        self.iteration += inc
        if not self._last_row_empty():
            self._save_last_row()
            self.table = np.concatenate([self.table, self.empty_row])
        self.table[-1]['iter'] = self.iteration

    def _save_last_row(self):
        text = "\n%d" % self.table[-1]['iter']
        for key in self.keys: text += ' %f' % self.table[-1][key]
        self.file.write(text)

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

    @staticmethod
    def plot(table:np.ndarray, keys:list=None, title:str='', logy=False):
        '''
        plot the table with specified keys
        if no keys are given, all keys are plotted
        '''
        plot = plt.semilogy if logy else plt.plot
        all_keys = list(table.dtype.fields)
        all_keys.remove('iter')
        if not keys: keys = all_keys
        else:
            for key in keys: assert key in all_keys
        for key in keys:
            y = table[key]
            idx = ~np.isnan(y)
            x = table['iter'][idx]
            y = y[idx]
            plot(x,y)
        plt.legend(keys)
        plt.title(title)
        plt.show()

    def save(self, filename:str):
        '''
        DEPRECATED
        save the logger result as a text file
        '''
        text = "%d" % self.iteration
        for key in self.keys: text += ' %s' % key
        for row in self.table:
            text += "\n%d" % row[0]
            for key in self.keys: text += ' %f' % row[key]
        with open(filename, 'w') as f:
            f.write(text)

    @classmethod
    def load(cls, filename:str) -> np.ndarray:
        '''
        loads the data array
        log file should look like

        key1 key2 key3 ...
        iter# key1_val key2_val key3_val ...
        iter# key1_val key2_val key3_val ...
        .
        .
        .

        '''
        with open(filename, 'r') as f:
            lines = f.read().split('\n')
        # remove empty lines
        while True:
            if lines[-1].split():
                break
            lines = lines[:-1]
            
        keys = lines[0].split()
        table = np.empty((len(lines)-1,), dtype=cls.dtype(keys))
        for i,line in enumerate(lines[1:]):
            values = line.split()
            if not values: break
            table[i]['iter'] = int(values[0])
            for j,key in enumerate(keys):
                table[i][key] = np.float64(values[j+1])
        return table

    @staticmethod
    def dtype(keys:list) -> np.dtype:
        '''
        return numpy datatype suitable for the keys
        '''
        types = [('iter', np.int64)]
        for key in keys:
            types.append((key, np.float64))
        return np.dtype(types)


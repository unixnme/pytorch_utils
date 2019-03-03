import unittest
from logger import Logger

class LoggerTest(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(keys=['train_acc', 'train_loss', 'val_acc', 'val_loss'],
                             desc='logger for unit test')

    def test_basic(self):
        for epoch in range(10):
            for i in range(100):
                self.logger.increment_iteration()
                self.logger.record('train_acc', self.logger.iteration * 0.1)
                self.logger.record('train_loss', 1. / self.logger.iteration)
            self.logger.record('val_acc', epoch)
            self.logger.record('val_loss', 1. / (epoch + 1))
        self.logger.plot()
        self.logger.save('log.txt')

    def test_load(self):
        logger = Logger.load('log.txt')
        logger.plot()
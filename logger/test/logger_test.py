import unittest
from logger import Logger
import os

class LoggerTest(unittest.TestCase):
    def test_basic(self):
        logger = Logger(keys=['train_acc', 'train_loss', 'val_acc', 'val_loss'],
                             desc='logger for unit test')
        for epoch in range(10):
            for i in range(100):
                logger.increment_iteration()
                logger.record('train_acc', logger.iteration * 0.1)
                logger.record('train_loss', 1. / logger.iteration)
            logger.record('val_acc', epoch)
            logger.record('val_loss', 1. / (epoch + 1))

        table = Logger.load('result.log')
        Logger.plot(table, logy=False)
        os.remove('result.log')
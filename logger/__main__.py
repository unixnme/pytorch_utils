from logger import Logger
import argparse

def plot():
    parser = argparse.ArgumentParser('Logger script')
    parser.add_argument('logfile', type=str, help='log file path to plot')
    parser.add_argument('--logy', action='store_true', help='plot in logy scale')
    args = parser.parse_args()

    table = Logger.load(args.logfile)
    Logger.plot(table, title=args.logfile, logy=args.logy)

plot()

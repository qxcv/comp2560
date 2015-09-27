#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import cycle
from sys import exit, stderr

from matplotlib import pyplot as plt
from pandas import read_csv

THRESH = 'Threshold'
MARKERS = ['<', 'o', '^', 'x', '>', '+', 'v', 's']

parser = ArgumentParser(
    description="Take PCK at different thresholds and plot it nicely"
)
parser.add_argument(
    '--outdir', type=str, default='graphs',
    help="Output directory to store graphs in"
)
parser.add_argument(
    '--outext', type=str, default='pgf', help="Extension for generated graphs"
)
parser.add_argument(
    '--show', default=False, action='store_true',
    help="Show the graphs instead of saving them"
)
parser.add_argument(
    '--input', nargs=2, metavar=('NAME', 'PATH'), action='append', default=[],
    help='Name (title) and path of CSV to plot; can be specified repeatedly'
)

def load_data(inputs):
    labels = []
    thresholds = None
    parts = None

    for name, path in inputs:
        labels.append(name)

        csv = read_csv(path)

        if thresholds is None:
            thresholds = csv[THRESH]
        if parts is None:
            parts = {part: [] for part in csv.columns - [THRESH]}

        assert len(parts) == len(csv.columns - [THRESH])
        assert (csv[THRESH] == thresholds).all()

        for part in parts:
            part_vals = csv[part]
            assert len(part_vals) == len(thresholds)
            parts[part].append(part_vals)

    return labels, thresholds, parts

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.input:
        parser.print_usage(stderr)
        print('error: must specify at least one --input', file=stderr)
        exit(1)

    labels, thresholds, parts = load_data(args.input)

    _, subplots = plt.subplots(1, len(parts), sharey=True)
    for part_name, subplot in zip(parts, subplots):
        pcks = parts[part_name]
        for pck, label, marker in zip(pcks, labels, cycle(MARKERS)):
            subplot.plot(thresholds, 100 * pck, label=label, marker=marker)
        subplot.set_title(part_name)
        subplot.set_xlabel('Threshold (px)')
        subplot.grid()

    subplots[0].set_ylabel('PCK (%)')
    subplots[0].set_ylim(ymin=0, ymax=100)
    subplots[0].set_yticks(range(0, 101, 10))

    plt.legend(labels, loc='lower right')

    if args.show:
        plt.show()
    else:
        raise NotImplementedError()

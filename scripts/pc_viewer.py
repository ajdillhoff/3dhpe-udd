import sys

import tables
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main(args):
    f = tables.open_file(args[1], mode='r')
    idx = int(args[2])

    s1 = f.root.predictions[idx]
    s2 = f.root.targets[idx]

    # Visualize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s1[:, 0], s1[:, 1], s1[:, 2])
    ax.scatter(s2[:, 0], s2[:, 1], s2[:, 2])
    plt.show()


if __name__ == "__main__":
    main(sys.argv)

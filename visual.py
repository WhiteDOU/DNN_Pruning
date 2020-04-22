import glob
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


root = sorted(glob.glob('./all_fc/*'))


def fc():
    i = 0
    name = 'fc' + str(i / 3)
    acc = root[i]
    run_time = root[i + 1]
    x = root[i + 2]
    with open(acc, 'r') as f:
        acc = f.readline()
        acc = acc.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(acc):
            acc[i] = float(acc[i])
    with open(x, 'r') as f:
        x = f.readline()
        x = x.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(x):
            x[i] = float(x[i])
    with open(run_time, 'r') as f:
        run_time = f.readline()
        run_time = run_time.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(run_time):
            run_time[i] = float(run_time[i])
    plt.figure()
    plt.title(name + 'ACC & RUN_TIME')
    plt.plot(x, acc, color='green', label='ACC')
    plt.plot(x, run_time, color='red', label='RUN_TIME')
    plt.xlabel('weights remain(%)')
    plt.ylabel('ACC & RUN_TIME(s) ')
    plt.legend()
    plt.show()
    i = 3
    name = 'fc' + str(i / 3)
    acc = root[i]
    run_time = root[i + 1]
    x = root[i + 2]
    with open(acc, 'r') as f:
        acc = f.readline()
        acc = acc.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(acc):
            acc[i] = float(acc[i])
    with open(x, 'r') as f:
        x = f.readline()
        x = x.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(x):
            x[i] = float(x[i])
    with open(run_time, 'r') as f:
        run_time = f.readline()
        run_time = run_time.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(run_time):
            run_time[i] = float(run_time[i])
    plt.figure()
    plt.title(name + 'ACC & RUN_TIME')
    plt.plot(x, acc, color='green', label='ACC')
    plt.plot(x, run_time, color='red', label='RUN_TIME')
    plt.xlabel('weights remain(%)')
    plt.ylabel('ACC & RUN_TIME(s) ')
    plt.legend()
    plt.show()


print(root)


def feature():
    for i in range(0, 15, 3):
        name = 'feature:' + str(i / 3)
        acc = root[i]
        x = root[i + 1]
        run_time = root[i + 2]
        with open(acc, 'r') as f:
            acc = f.readline()
            acc = acc.replace(' ', '').replace('[', '').replace(']', '').split(',')
            for i, item in enumerate(acc):
                acc[i] = float(acc[i])
        with open(x, 'r') as f:
            x = f.readline()
            x = x.replace(' ', '').replace('[', '').replace(']', '').split(',')
            for i, item in enumerate(x):
                x[i] = float(x[i])
        with open(run_time, 'r') as f:
            run_time = f.readline()
            run_time = run_time.replace(' ', '').replace('[', '').replace(']', '').split(',')
            for i, item in enumerate(run_time):
                run_time[i] = float(run_time[i])
        plt.figure()
        plt.title(name + 'ACC & RUN_TIME')
        plt.plot(x, acc, color='green', label='ACC')
        plt.plot(x, run_time, color='red', label='RUN_TIME')
        plt.xlabel('weights remain(%)')
        plt.ylabel('ACC & RUN_TIME(s) ')
        plt.legend()
        plt.show()


def fc_all():
    acc = root[0]
    coord = root[1]
    run_time = root[2]
    with open(acc, 'r') as f:
        acc = f.readline()
        acc = acc.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(acc):
            acc[i] = float(acc[i])
    with open(coord, 'r') as f:
        coord = f.readline()
        coord = coord.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(coord):
            coord[i] = float(coord[i])
        x = coord[0::2]
        y = coord[1::2]
    with open(run_time, 'r') as f:
        run_time = f.readline()
        run_time = run_time.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i, item in enumerate(run_time):
            run_time[i] = float(run_time[i])
    x = np.linspace(14, 0.0025, num=100)
    y = np.linspace(5, 0.0025, num=100)
    acc = np.array(acc).reshape(x.shape[0],y.shape[0])
    run_time = np.array(run_time).reshape(x.shape[0],y.shape[0])
    print(run_time)
    fig = plt.figure()
    ax = Axes3D(fig)
    x,y = np.meshgrid(x,y)
    ax.plot_surface(x,y,acc,rstride=1, cstride=1, cmap='rainbow')
    plt.xlabel('fc0')
    plt.ylabel('fc1')
    plt.show()
    plt.xlabel('fc0')
    plt.ylabel('fc1')
    ax.plot_surface(x,y,run_time,rstride=1, cstride=1, cmap='rainbow')
    plt.show()



fc_all()


import matplotlib.pyplot as plt

def graph_trajectory_xyz(px, py, pz, pgrp):
    graph_axis_trajectory(px, 'x')
    graph_axis_trajectory(py, 'y')
    graph_axis_trajectory(pz, 'z')
    graph_axis_trajectory(pgrp, 'grp')

def graph_trajectory_joint(pth1, pth2, pth3):
    graph_axis_trajectory(pth1, 'th1')
    graph_axis_trajectory(pth2, 'th2')
    graph_axis_trajectory(pth3, 'th3')

def graph_axis_trajectory(axis, pdf_name):
    time = []
    pos = []
    vel = []
    acc = []
    for t in axis:
        time.append(t[0])
        pos.append(t[1])
        vel.append(t[2])
        acc.append(t[3])

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(311)
    ax1.plot(time, pos)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax2 = fig.add_subplot(312)
    ax2.plot(time, vel)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax3 = fig.add_subplot(313)
    ax3.plot(time, acc)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s^2)')
    fig.savefig((pdf_name + ".pdf"))

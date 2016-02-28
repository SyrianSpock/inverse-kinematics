import pygame, sys, time
import matplotlib.pyplot as plt
from pickit.Datatypes import *
from pickit import DebraArm, ArmManager
from math import cos, sin, pi
from pygame_utils import *
from graph_utils import *

# Robot settings
L1 = 1.5
L2 = 1.0
L3 = 0.2
GRIPPER_HEADING = 0
RANGE_MIN = abs(L1 - L2)
RANGE_MAX = abs(L1 + L2)

# Trajectory generation settings
DELTA_T = 0.05

# Display settings
PX_PER_METER = 100
WIDTH = int(3 * (L1 + L2 + L3) * PX_PER_METER)
HEIGHT = int(3 * (L1 + L2 + L3) * PX_PER_METER)

pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
PURPLE = (255, 0, 255)

def main():
    "draw loop"

    # Initial robot state
    origin = Vector3D(-0.5, 0.0, 0.0)

    arm = DebraArm.DebraArm(l1=L1, l2=L2, origin=origin, flip_x=-1)
    arm.inverse_kinematics(RobotSpacePoint(-0.99*(L1+L2)+origin.x, 0+origin.y, 0+origin.z, 0))
    tool = arm.get_tool()
    joints = arm.get_joints()

    ws_front = Workspace(-1.5 + origin.x,
                         1.5 + origin.x,
                         abs(L1 - L2) + origin.y,
                         abs(L1 + L2) + origin.y,
                         0.0 + origin.z,
                         0.2 + origin.z,
                         1)
    ws_side = Workspace(-abs(L1 + L2) + origin.x,
                        -abs(L1 - L2) + origin.x,
                        -1.5 + origin.y,
                        1.5 + origin.y,
                        0.0 + origin.z,
                        0.2 + origin.z,
                        1)
    ws_back = Workspace(-1.5 + origin.x,
                        1.5 + + origin.x,
                        -abs(L1 + L2) + origin.y,
                        -abs(L1 - L2) + origin.y,
                        0.0 + origin.z,
                        0.2 + origin.y,
                        -1)
    arm_manager = ArmManager.ArmManager(arm, ws_front, ws_side, ws_back, DELTA_T)

    # Draw robot
    origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
    draw_workspaces(ws_front, ws_side, ws_back)

    pygame.display.update()

    paused = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = get_cursor_pos()
                tool_prev = arm_manager.arm.get_tool()
                tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)

                start_time = time.time()
                pth1, pth2, pz, pth3 = arm_manager.goto(tool_prev, RobotSpacePoint(0,0,0,0),
                                                        tool, RobotSpacePoint(0,0,0,0),
                                                        'line')
                elapsed_time = time.time() - start_time
                print('elapsed time: ', elapsed_time)

                graph_trajectory_joint(pth1, pth2, pth3)
                draw_trajectory(arm_manager.arm, pth1, pth2, pz, pth3, DELTA_T)

        if not paused:
            SCREEN.fill(BLACK)

            origin, p1, p2, p3, z = arm_manager.arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
            draw_workspaces(ws_front, ws_side, ws_back)

            pygame.display.update()


if __name__ == "__main__":
    main()

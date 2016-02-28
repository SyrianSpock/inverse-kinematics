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
    ALTERNATE = 0

    # Initial robot state
    right_origin = Vector3D(0.5, 0.0, 0.0)
    left_origin = Vector3D(-0.5, 0.0, 0.0)

    right_arm = DebraArm.DebraArm(l1=L1, l2=L2, origin=right_origin, flip_x=1)
    right_arm.inverse_kinematics(RobotSpacePoint(0.99*(L1+L2) + right_origin.x,
                                                 0 + right_origin.y,
                                                 0 + right_origin.z,
                                                 0))
    tool = right_arm.get_tool()
    joints = right_arm.get_joints()

    left_arm = DebraArm.DebraArm(l1=L1, l2=L2, origin=left_origin, flip_x=-1)
    left_arm.inverse_kinematics(RobotSpacePoint(-0.99*(L1+L2) + left_origin.x,
                                                0 + left_origin.y,
                                                0 + left_origin.z,
                                                0))
    tool = left_arm.get_tool()
    joints = left_arm.get_joints()

    ws_front = Workspace(-1.5, 1.5,
                         abs(L1 - L2), abs(L1 + L2),
                         0.0, 0.2,
                         1)
    ws_back = Workspace(-1.5, 1.5,
                        -abs(L1 + L2), -abs(L1 - L2),
                        0.0, 0.2,
                        -1)
    ws_right = Workspace(abs(L1 - L2) + right_origin.x, abs(L1 + L2) + right_origin.x,
                        -1.5 + right_origin.y, 1.5 + right_origin.y,
                        0.0 + right_origin.z, 0.2 + right_origin.z,
                        1)
    ws_left = Workspace(-abs(L1 + L2) + left_origin.x, -abs(L1 - L2) + left_origin.x,
                        -1.5 + left_origin.y, 1.5 + left_origin.y,
                        0.0 + left_origin.z, 0.2 + left_origin.z,
                        1)

    right_arm_manager = ArmManager.ArmManager(right_arm, ws_front, ws_right, ws_back, DELTA_T)
    left_arm_manager = ArmManager.ArmManager(left_arm, ws_front, ws_left, ws_back, DELTA_T)

    # Draw right arm
    origin, p1, p2, p3, z = right_arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
    draw_workspaces(ws_front, ws_right, ws_back)
    # Draw left arm
    origin, p1, p2, p3, z = left_arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
    draw_workspaces(ws_front, ws_left, ws_back)

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
                if ALTERNATE:
                    ALTERNATE = 0
                    tool_prev = right_arm_manager.arm.get_tool()
                    tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)

                    start_time = time.time()
                    pth1, pth2, pz, pth3 = right_arm_manager.goto(tool_prev, RobotSpacePoint(0,0,0,0),
                                                                  tool, RobotSpacePoint(0,0,0,0),
                                                                  'line')
                    elapsed_time = time.time() - start_time
                    print('elapsed time: ', elapsed_time)

                    graph_trajectory_joint(pth1, pth2, pth3)
                    draw_trajectory(right_arm_manager.arm, pth1, pth2, pz, pth3, DELTA_T)

                else:
                    ALTERNATE = 1
                    tool_prev = left_arm_manager.arm.get_tool()
                    tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)

                    start_time = time.time()
                    pth1, pth2, pz, pth3 = left_arm_manager.goto(tool_prev, RobotSpacePoint(0,0,0,0),
                                                                 tool, RobotSpacePoint(0,0,0,0),
                                                                 'line')
                    elapsed_time = time.time() - start_time
                    print('elapsed time: ', elapsed_time)

                    graph_trajectory_joint(pth1, pth2, pth3)
                    draw_trajectory(left_arm_manager.arm, pth1, pth2, pz, pth3, DELTA_T)

        if not paused:
            SCREEN.fill(BLACK)

            # Draw right arm
            origin, p1, p2, p3, z = right_arm_manager.arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
            draw_workspaces(ws_front, ws_right, ws_back)
            # Draw left arm
            origin, p1, p2, p3, z = left_arm_manager.arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
            draw_workspaces(ws_front, ws_left, ws_back)

            pygame.display.update()

if __name__ == "__main__":
    main()

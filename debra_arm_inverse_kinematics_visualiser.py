import pygame, sys
from pickit.Datatypes import *
from pickit import DebraArm
from math import cos, sin, pi
from pygame_utils import *

PX_PER_METER = 100
L1 = 1.5
L2 = 1.0
L3 = 0.2
GRIPPER_HEADING = 0
WIDTH = int(2 * (L1 + L2 + L3) * PX_PER_METER)
HEIGHT = int(2 * (L1 + L2 + L3) * PX_PER_METER)

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
    origin_x, origin_y = 0.0, 0.0

    arm = DebraArm.DebraArm(l1=L1, l2=L2, flip_x=-1, flip_elbow=1)
    tool = arm.get_tool()
    joints = arm.get_joints()

    # Draw robot
    origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3)

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

        if not paused:
            SCREEN.fill(BLACK)

            (x, y) = pygame.mouse.get_pos()
            x = (x - WIDTH/2) / PX_PER_METER
            y = - (y - HEIGHT/2) / PX_PER_METER
            print("cursor: x: ", x, ", y: ", y)

            try:
                tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)
                joints = arm.inverse_kinematics(tool)
            except ValueError:
                pass

            print("arm: ", "x:", arm.tool.x, "y:", arm.tool.y, \
                  "th1:", arm.joints.theta1, "th2:", arm.joints.theta2)

            # Draw robot
            origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3)

            pygame.display.update()

if __name__ == "__main__":
    main()

import pygame, sys, time
from invkin.Datatypes import *
from invkin.PathPlanner import *
from invkin import DebraArm
from math import cos, sin, pi

# Robot settings
L1 = 1.5
L2 = 1.0
L3 = 0.2
GRIPPER_HEADING = 0
RANGE_MIN = abs(L1 - L2)
RANGE_MAX = abs(L1 + L2)

# Trajectory generation settings
PLAN_ROBOT_SPACE = 1
PLAN_JOINT_SPACE = 2
MODE = PLAN_ROBOT_SPACE
DURATION = 1.0
DELTA_T = 0.05

# Display settings
PX_PER_METER = 100
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

    arm = DebraArm.DebraArm(l1=L1, l2=L2, flip_x=-1, flip_elbow=-1)
    tool = arm.forward_kinematics()
    joints = arm.inverse_kinematics()

    # Draw robot
    origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)

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
                tool_prev = tool
                tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)

                if MODE == PLAN_JOINT_SPACE:
                    path_th1, path_th2, path_z, path_th3 = \
                        get_path_joint_space(arm,
                                             tool_prev,
                                             RobotSpacePoint(0,0,0,0),
                                             tool,
                                             RobotSpacePoint(0,0,0,0),
                                             DURATION,
                                             DELTA_T)
                elif MODE == PLAN_ROBOT_SPACE:
                    path_th1, path_th2, path_z, path_th3 = \
                        get_path_robot_space(arm,
                                             tool_prev,
                                             RobotSpacePoint(0,0,0,0),
                                             tool,
                                             RobotSpacePoint(0,0,0,0),
                                             DURATION,
                                             DELTA_T)

                draw_trajectory(arm, path_th1, path_th2, path_z, path_th3, DELTA_T)

        if not paused:
            SCREEN.fill(BLACK)

            origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)

            pygame.display.update()

def draw_trajectory(arm, path_th1, path_th2, path_z, path_th3, dt):
    "draw trajectory"
    for th1, th2, z, th3 in zip(path_th1, path_th2, path_z, path_th3):
        joints = JointSpacePoint(th1[1], th2[1], z[1], th3[1])
        tool = arm.update_joints(joints)
        get_robot_new_state(arm, tool)

        print("arm: ", "x:", arm.tool.x, "y:", arm.tool.y, \
              "th1:", arm.joints.theta1, "th2:", arm.joints.theta2)

        # Draw robot
        origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
        draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)

        pygame.display.update()

        time.sleep(dt)

def get_robot_new_state(arm, new_tool):
    "get robot's current state"
    try:
        return arm.update_tool(new_tool)
    except ValueError:
        return arm.inverse_kinematics()

def get_cursor_pos():
    "get cursor position"
    (x, y) = pygame.mouse.get_pos()
    x = (x - WIDTH/2) / PX_PER_METER
    y = - (y - HEIGHT/2) / PX_PER_METER
    print("cursor: x: ", x, ", y: ", y)

    return x, y

def draw_arm(p0, p1, p2, p3, RANGE_MIN, RANGE_MAX):
    "draw arm state"

    draw_line(p0.x, p0.y, p1.x, p1.y)
    draw_line(p1.x, p1.y, p2.x, p2.y)
    draw_line(p2.x, p2.y, p3.x, p3.y)
    draw_circle(p0.x, p0.y, RANGE_MAX)
    if RANGE_MIN * PX_PER_METER > 1:
        draw_circle(p0.x, p0.y, RANGE_MIN)

def draw_line(pos1_x, pos1_y, pos2_x, pos2_y):
    "draw line from pos1 to pos2"
    pygame.draw.line(SCREEN,
                     CYAN,
                     (int(pos1_x * PX_PER_METER + WIDTH/2),
                      int(-pos1_y * PX_PER_METER + HEIGHT/2)),
                     (int(pos2_x * PX_PER_METER + WIDTH/2),
                      int(-pos2_y * PX_PER_METER + HEIGHT/2)),
                     2)

def draw_circle(pos_x, pos_y, radius):
    "draw circle from center position and radius"
    pygame.draw.circle(SCREEN,
                       RED,
                       (int(pos_x * PX_PER_METER + WIDTH/2),
                        int(-pos_y * PX_PER_METER + HEIGHT/2)),
                       int(radius * PX_PER_METER),
                       2)

if __name__ == "__main__":
    main()

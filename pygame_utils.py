from pickit.Datatypes import *
from pickit import DebraArm, ArmManager
import pygame

def draw_workspaces(ws_1, ws_2, ws_3):
    "draw workspaces"
    draw_workspace(ws_1)
    draw_workspace(ws_2)
    draw_workspace(ws_3)

def draw_workspace(workspace):
    "draw workspace"
    draw_line(workspace.x_min, workspace.y_min, workspace.x_max, workspace.y_min, GREEN)
    draw_line(workspace.x_max, workspace.y_min, workspace.x_max, workspace.y_max, GREEN)
    draw_line(workspace.x_max, workspace.y_max, workspace.x_min, workspace.y_max, GREEN)
    draw_line(workspace.x_min, workspace.y_max, workspace.x_min, workspace.y_min, GREEN)

def draw_arm(p0, p1, p2, p3, RANGE_MIN, RANGE_MAX):
    "draw arm state"

    draw_line(p0.x, p0.y, p1.x, p1.y, CYAN)
    draw_line(p1.x, p1.y, p2.x, p2.y, CYAN)
    draw_line(p2.x, p2.y, p3.x, p3.y, CYAN)
    draw_circle(p0.x, p0.y, RANGE_MAX)
    if RANGE_MIN * PX_PER_METER > 1:
        draw_circle(p0.x, p0.y, RANGE_MIN)

def draw_scara(p0, p1, p2):
    "draw scara state"

    draw_line(p0.x, p0.y, p1.x, p1.y)
    draw_line(p1.x, p1.y, p2.x, p2.y)

def draw_line(pos1_x, pos1_y, pos2_x, pos2_y, color):
    "draw line from pos1 to pos2"
    pygame.draw.line(SCREEN,
                     color,
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

def get_cursor_pos():
    "get cursor position"
    (x, y) = pygame.mouse.get_pos()
    x = (x - WIDTH/2) / PX_PER_METER
    y = - (y - HEIGHT/2) / PX_PER_METER
    print("cursor: x: ", x, ", y: ", y)

    return x, y

def draw_trajectory(arm, path_th1, path_th2, path_z, path_th3, dt):
    "draw trajectory"

    for th1, th2, z, th3 in zip(path_th1, path_th2, path_z, path_th3):
        joints = JointSpacePoint(th1[1], th2[1], z[1], th3[1])
        tool = arm.forward_kinematics(joints)
        get_robot_new_state(arm, tool)

        # Draw robot
        origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
        draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)

        pygame.display.update()

        time.sleep(dt)

def get_robot_new_state(arm, new_tool):
    "get robot's current state"
    try:
        return arm.inverse_kinematics(new_tool)
    except ValueError:
        return arm.get_joints()

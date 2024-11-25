from math import sqrt, atan2, pi

def pyth(point1, point2): # distance between two points
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def cal_direction(point1, point2): # direction from point1 to point2
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0
    else:
        return atan2(point2[1] - point1[1], point2[0] - point1[0])

def cal_direction_change(s1, s2): # direction change from s1 to s2
    direction_change = s2 - s1
    if direction_change > pi:
        return direction_change - pi * 2
    elif direction_change <= -pi:
        return direction_change + pi * 2
    else:
        return direction_change

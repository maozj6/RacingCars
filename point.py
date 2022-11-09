class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)


def IsPointInMatrix(p1, p2, p3, p4, p):
    isPointIn = GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0
    return isPointIn


if __name__ == '__main__':
    p1 = Point(213, -54)
    p2 = Point(226, 1)
    p3 = Point(4, 4)
    p4 = Point(1, 3)

    pp = Point(2, 2)
    pp2 = Point(4, 2)

    print(IsPointInMatrix(p1, p2, p3, p4, pp))

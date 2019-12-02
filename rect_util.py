# Minimum distance to two detected bounding boxes to be considered differents
min_rec_dist = 50

def superposition(rec1, rec2):
    (x1, y1, w1, h1) = rec1
    (x2, y2, w2, h2) = rec2

    return (x1 < x2 and x1 + w1 >= x2 and y1 < y2 and y1 + h1 >= y2) or \
        (x1 < x2 and x1 + w1 >= x2 and y1 >= y2 and y1 < y2 + h2) or \
        (x1 >= x2 and x1 < x2 + w2 and y1 < y2 and y1 + h1 >= y2) or \
        (x1 >= x2 and x1 < x2 + w2 and y1 >= y2 and y1 < y2 + h2)


def euclidian_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def rec_distance(rec1, rec2):
    (x1, y1, w1, h1) = rec1
    (x2, y2, w2, h2) = rec2

    if superposition(rec1, rec2):
        return 0
    else:
        c1 = (x1 + w1/2, y1 + h1/2)
        c2 = (x2 + w2/2, y2 + h2/2)
        dx = abs(c1[0] - c2[0]) - w1/2 - w2/2
        dy = abs(c1[1] - c2[1]) - h1/2 - h2/2

        return max(dx, dy)

def merge_rects(rec1, rec2):
    (x1, y1, w1, h1) = rec1
    (x2, y2, w2, h2) = rec2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


# Avoid superpositions and rectangles too close
# This code will check with all pairs of rectangles if they are too close
# When a pair that is too close is found the rectangles are merged and
# a recursive call is made with a reduced list of rectangles removing the previous
# pair and adding the merged result

def adjust_rectangles(rectangles):
    final_rects = []
    modifieds = []
    for i in range(len(rectangles)):
        for j in range(i+1, len(rectangles)):
            r0 = rectangles[i]
            r1 = rectangles[j]
            if rec_distance(r0, r1) < min_rec_dist:
                rec = merge_rects(r0, r1)
                rectangles.remove(r0)
                rectangles.remove(r1)
                rectangles.append(rec)
                return adjust_rectangles(rectangles)

    return rectangles

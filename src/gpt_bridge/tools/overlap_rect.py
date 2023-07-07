def check_overlap(box1, box2):
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return (False,-1)
    else:
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        overlap_area = x_overlap * y_overlap
        return (True, overlap_area)

# Input coordinates for the two boxes xyxy
box1 = [0, 0, 8, 8]
box2 = [3, 3, 8, 8]

a,b=check_overlap(box1, box2)
print(a,b)
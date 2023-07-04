import matplotlib.pyplot as plt

def plot_points(coordinates):
    lines = [[0, 1], [1, 2], [0, 3], [3, 4], [4, 5]]
    x = [point[0] for point in coordinates]
    y = [point[1] for point in coordinates]

    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Points')

    # Add index labels
    for i, point in enumerate(coordinates):
        plt.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')

    # Draw lines
    for line in lines:
        x_line = [coordinates[line[0]][0], coordinates[line[1]][0]]
        y_line = [coordinates[line[0]][1], coordinates[line[1]][1]]
        plt.plot(x_line, y_line, 'k-')
    plt.axis('equal')
    plt.show()

def plot_limbs(coordinates):
    lines = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    x = coordinates[0]
    y = coordinates[1]

    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Points')

    # Add index labels
    # for i, point in enumerate(coordinates):
    #     plt.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')

    # Draw lines
    for line in lines:
        x_line = [coordinates[0][line[0]-1], coordinates[0][line[1]-1]]
        y_line = [coordinates[1][line[0]-1], coordinates[1][line[1]-1]]
        plt.plot(x_line, y_line, 'k-')
    plt.axis('equal')
    plt.show()


#a = [[10, 92], [1, 83], [4, 89], [29, 92], [35, 78], [34, 70] ]
a = [
(16, 95), (9, 79), (4, 68), (36, 93), (40, 75), (41, 61)
]
plot_points(a)

limb =[[23, 23, 23, 21, 31, 16, 36, 9, 40, 4, 41, 20, 33, 18, 31, 21, 30], [107, 109, 109, 109, 108, 95, 93, 79, 75, 68, 61, 61, 60, 37, 34, 13, 7]]
#plot_limbs(limb)

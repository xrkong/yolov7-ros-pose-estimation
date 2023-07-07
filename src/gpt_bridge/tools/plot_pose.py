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
[16, 98], [10, 80], [4, 64], [34, 102], [43, 91], [44, 84]
]
plot_points(a)

limb =[[18, 18, 18, 17, 27, 16, 34, 10, 43, 4, 44, 23, 35, 28, 39, 31, 47], [110, 112, 112, 111, 113, 98, 102, 80, 91, 64, 84, 63, 65, 34, 40, 8, 21]]
plot_limbs(limb)

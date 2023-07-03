import matplotlib.pyplot as plt

points = [(95, 190), (90, 180), (85, 170), (105, 190), (110, 180), (115, 170)]

x = [point[0] for point in points]
y = [point[1] for point in points]

plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Points')

# Marking the indices
for i, point in enumerate(points):
    plt.text(point[0], point[1], str(i), color='red')

plt.grid(True)
plt.show()
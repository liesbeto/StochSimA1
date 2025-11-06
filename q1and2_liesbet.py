import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

def randomsample_1d(upper, lower):
    return np.random.random() * (upper-lower) + lower

def randomsample_3d(upper, lower):
    """Creates a random 3D point within a cube with bounds upper and lower."""
    x = randomsample_1d(upper, lower)
    y = randomsample_1d(upper, lower)
    z = randomsample_1d(upper, lower)

    return x, y, z

def logisticsample_1d(current, m=3.8):
    return m*current * (1-current)

def logisticsample_3d(coords):
    x, y, z = coords
    x = logisticsample_1d(x)
    y = logisticsample_1d(y)
    z = logisticsample_1d(z)
    coords = x, y, z
    return coords

def in_sphere(coord, k):
    x, y, z = coord
    return x**2 + y**2 + z**2 <= k**2

def in_torus(coord, r_maj, r_min):
    x, y, z = coord
    return (np.sqrt(x**2 + y**2) - r_maj)**2 + z**2 <= r_min**2

def visualise_dots(list_coords, upper, lower):
    lx, ly, lz = [], [], []
    for x,y,z in list_coords:
        lx.append(x)
        ly.append(y)
        lz.append(z)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(lx, ly, lz, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    plt.show()

def visualise_sphere(list_coords, k, upper, lower):
    lx, ly, lz = [], [], []
    for x,y,z in list_coords:
        if in_sphere((x,y,z), k):
            lx.append(x)
            ly.append(y)
            lz.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(lx, ly, lz, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    plt.show()

def visualise_torus(list_coords, r_maj, r_min, upper, lower):
    lx, ly, lz = [], [], []
    for x,y,z in list_coords:
        if in_torus((x,y,z), r_maj, r_min):
            lx.append(x)
            ly.append(y)
            lz.append(z)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(lx, ly, lz, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    plt.show()

    plt.scatter(lx, ly, s=1)
    plt.title("Top View of Samples in Torus")
    plt.show()

def visualise_intersection(list_coords, k, r_maj, r_min, upper, lower):
    lx, ly, lz = [], [], []
    for x,y,z in list_coords:
        if in_sphere((x,y,z), k) and in_torus((x,y,z), r_maj, r_min):
            lx.append(x)
            ly.append(y)
            lz.append(z)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(lx, ly, lz, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    plt.show()

def randomsequence_1d(upper, lower, steps):
    return [randomsample_1d(upper, lower) for _ in range(steps)]

def randomsequence_3d(upper, lower, steps):
    list_coords = []
    for _ in range(steps):
        coords = randomsample_3d(upper, lower)
        list_coords.append(coords)
    
    return list_coords

def random_intersection(N, upper, lower, k, r_maj, r_min, visualise=False):
    list_intersect = []

    list_coords = randomsequence_3d(upper, lower, N)
    for coords in list_coords:
        if in_sphere(coords, k) and in_torus(coords, r_maj, r_min):
            list_intersect.append(1)
    
    intersect_frac = len(list_intersect) / N
    intersect_size = intersect_frac * ((upper - lower) ** 3)

    if visualise:
        visualise_sphere(list_coords, k, upper, lower)
        visualise_torus(list_coords, r_maj, r_min, upper, lower)
        visualise_intersection(list_coords, k, r_maj, r_min, upper, lower)

    return intersect_size

def transform_to_bounds(value, upper, lower, mini=0.1805, maxi=0.95):
    return (((value - mini) * (upper - lower)) / (maxi - mini)) + lower

def logisticsequence_1d(x_0, upper, lower, steps):
    x = x_0
    lx = [x]
    while len(lx) < steps:
        x = logisticsample_1d(x)
        lx.append(x)
    
    for i in range(steps):
        lx[i] = transform_to_bounds(lx[i], upper, lower)

    return lx

def logisticsequence_3d(upper, lower, steps):
    coords_0=(np.random.random()*(0.95-0.1805)*0.1805, np.random.random()*(0.95-0.1805)*0.1805, np.random.random()*(0.95-0.1805)*0.1805)
    x, y, z = coords_0
    lx, ly, lz = [x], [y], [z]

    for _ in range(steps-1):
        x, y, z = logisticsample_3d((x, y, z))
        lx.append(x)
        ly.append(y)
        lz.append(z)

    for i in range(steps):
        lx[i] = transform_to_bounds(lx[i], upper, lower)
        ly[i] = transform_to_bounds(ly[i], upper, lower)
        lz[i] = transform_to_bounds(lz[i], upper, lower)

    list_coords = []
    for i in range(len(lx)):
        list_coords.append((lx[i], ly[i], lz[i]))

    return list_coords

def logistic_intersection(N, upper, lower, k, r_maj, r_min, visualise=False):
    list_intersect = []

    list_coords = logisticsequence_3d(upper, lower, N)
    for coords in list_coords:
        if in_sphere(coords, k) and in_torus(coords, r_maj, r_min):
            list_intersect.append(1)

    intersect_frac = len(list_intersect) / N
    intersect_size = intersect_frac * ((upper - lower) ** 3)

    if visualise:
        visualise_dots(list_coords, upper, lower)
        visualise_sphere(list_coords, k, upper, lower)
        visualise_torus(list_coords, r_maj, r_min, upper, lower)
        visualise_intersection(list_coords, k, r_maj, r_min, upper, lower)

    return intersect_size

def plot_distribution(func, upper, steps):
    lower = -1*upper
    if func == logisticsequence_1d:
        lx = logisticsequence_1d(np.random.random()*(0.95-0.1805)+0.1805, upper, lower, steps)
        title = "Logistic Sequence Samples Distribution"
    else:
        lx = randomsequence_1d(upper, lower, steps)
        title = "Uniform Random Samples Distribution"

    plt.hist(lx, bins=100)
    plt.title(title)
    plt.xlabel("Sample value")
    plt.ylabel("Frequency")
    plt.show()

def plot_comparison(case, bounds):
    """
    Plots the random uniform distribution and logistic distribution for a list
    of bounds.
    """ 
    k = 1
    if case == "A":
        r_maj = 0.75
        r_min = 0.4
    elif case == "B":
        r_maj = 0.5
        r_min = 0.5
    else:
        print('Error: Case must be "A" or "B"')
        return

    uniform_randoms, uniform_randoms_err = [], []
    logistics, logistics_err = [], []
    for bound in bounds:
        uni_temp = []
        log_temp = []
        for _ in range(10):
            uni_temp.append(random_intersection(100000, bound, -1*bound, k, r_maj, r_min))
            log_temp.append(logistic_intersection(100000, bound, -1*bound, k, r_maj, r_min))
        uniform_randoms.append(np.mean(uni_temp))
        uniform_randoms_err.append(np.std(uni_temp))
        logistics.append(np.mean(log_temp))
        logistics_err.append(np.std(log_temp))

    plt.errorbar(bounds, uniform_randoms, uniform_randoms_err, label="Uniform")
    plt.errorbar(bounds, logistics, logistics_err, label="Logistic")
    plt.legend()
    plt.title("Intersection Area vs Bounding Box")
    plt.ylabel("Intersection Area")
    plt.xlabel("Bounding Box Upper Bound")
    plt.show()

plot_distribution(logisticsequence_1d, 1.2, 100000)
plot_distribution(randomsequence_1d, 1.2, 100000)
# plot_distribution(logisticsequence_1d, 3.0, 100000)
# plot_distribution(randomsequence_1d, 3.0, 100000)

# the results of the logistic intersection depend on the bounds
print(logistic_intersection(100000, 1.2, -1.2, 1, 0.75, 0.4, visualise=True))
# print(logistic_intersection(100000, 3, -3, 1, 0.75, 0.4, visualise=True))
# print(logistic_intersection(100000, 1.15, -1.15, 1, 0.5, 0.5))

plot_comparison("A", [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])

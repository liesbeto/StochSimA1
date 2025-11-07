import numpy as np
import matplotlib.pyplot as plt
import statistics as st
np.random.seed(0)

def montecarlo_point(upper, lower, P):
    """Creates a random 3D point within a cube with bounds upper and lower."""
    if np.random.random() < P:
        x = np.random.random() * (upper-lower) + lower
        y = np.random.random() * (upper-lower) + lower
        z = np.random.random() * (upper-lower) + lower
        small = False
    else:
        x = np.random.random() * (1.151-(-1.151)) + (-1.151)
        y = np.random.random() * (1.151-(-1.151)) + (-1.151)
        z = np.random.random() * (0.501-(-0.301)) + (-0.301)
        small = True
    return (x, y, z), small

def logistic_sequence(coords_n):
    x, y, z = coords_n

def in_sphere(coord, k):
    x, y, z = coord
    return x**2 + y**2 + z**2 <= k**2

def in_torus(coord, r_maj, r_min, center_coord):
    x, y, z = coord
    x_c, y_c, z_c = center_coord
    return (np.sqrt((x-x_c)**2 + (y-y_c)**2) - r_maj)**2 + (z-z_c)**2 <= r_min**2

def visualise_all(list_coords, k):
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
    plt.show()

def visualise_sphere(list_coords, k):
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
    plt.show()

def visualise_torus(list_coords, r_maj, r_min, center_coord):
    lx, ly, lz = [], [], []
    for x,y,z in list_coords:
        if in_torus((x,y,z), r_maj, r_min, center_coord):
            lx.append(x)
            ly.append(y)
            lz.append(z)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(lx, ly, lz, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    plt.scatter(lx, ly)
    plt.show()


def visualise_intersection(list_coords, k, r_maj, r_min):
    lx, ly, lz = [], [], []
    for x,y,z in list_coords:
        if in_sphere((x,y,z), k) and in_torus((x,y,z), r_maj, r_min, center_coord):
            lx.append(x)
            ly.append(y)
            lz.append(z)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(lx, ly, lz, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def calculate_intersection_boxes(N, upper, lower, k, r_maj, r_min, sampling_method="montecarlo", coords_0 = None, center_coord=[0,0,0], P=1):
    intersect_inside_bigbox = 0
    intersect_inside_smallbox = 0
    N_small=0
    N_big=0
    list1 = []
    list_coords = []
    just_started = True
    for _ in range(N):
        if sampling_method == "montecarlo":
            coords, small = montecarlo_point(upper, lower, P)
        else:
            if just_started:
                coords = logistic_sequence(coords_0)
            else:
                coords = logistic_sequence(list_coords[-1])
        list_coords.append(coords)
        if small:
            if in_sphere(coords, k) and in_torus(coords, r_maj, r_min, center_coord):
                intersect_inside_smallbox += 1
            N_small += 1

        elif in_sphere(coords, k) and in_torus(coords, r_maj, r_min, center_coord):
            intersect_inside_bigbox +=1
            N_big += 1
        else:
            N_big += 1
        
        if in_sphere(coords, k) and in_torus(coords, r_maj, r_min, center_coord):
            list1.append(1)


    
    intersect_frac = len(list1) / N
    intersect_size = intersect_frac * ((upper - lower) ** 3)
    if N_small > 0:
        intersect_frac_small = intersect_inside_smallbox / N_small
        intersect_size_small = intersect_frac_small * ((0.501-(-0.301)) * (1.151 - (-1.151)) ** 2 )
    if N_big > 0:
        intersect_frac_big = intersect_inside_bigbox / N_big
        intersect_size_big = intersect_frac_big * ((upper - lower) ** 3)    
#    visualise_all(list_coords, k)
#    visualise_sphere(list_coords, k)
#    visualise_torus(list_coords, r_maj, r_min, center_coord)
#    visualise_intersection(list_coords, k, r_maj, r_min)
    if N_small > 0 and N_big == 0:
        return intersect_size_small
    if N_big > 0 and N_small == 0:
        return intersect_size_big
    else:
        return intersect_size_small, intersect_size_big

#torus height = z + - 0.4
center_coord=[0, 0, 0.1]
#print(calculate_intersection(100000, 1.15, -1.15, 1, 0.75, 0.4))
print(calculate_intersection_boxes(100000, 1.50, -1.15, 1, 0.75, 0.4, center_coord= center_coord, P=0.5))
#print(calculate_intersection(100000, 1.15, -1.15, 1, 0.5, 0.5))

values_list = []
for prob in range(1,10):
    p = prob/10
    temp_values = []
    for _ in range(100):
        values = calculate_intersection_boxes(100000, 1.50, -1.15, 1, 0.75, 0.4, center_coord= center_coord, P=p)
        temp_values.append(values)
    big_vals = [x[1] for x in temp_values]
    small_vals = [x[0] for x in temp_values]
    values_list.append(((p, np.mean(small_vals), np.std(small_vals)), (p, np.mean(big_vals), np.std(big_vals))))

plotline_list = [[[],[],[]],[[],[],[]]] # list of list of small and list of big values, and then a list of all p, mu, sigma in each values list
for values_small, values_big in values_list:
    p, mean, stdev = values_small
    p = np.array(p)
    p = p - 0.005
    plotline_list[0][0].append(p)
    plotline_list[0][1].append(mean)
    plotline_list[0][2].append(stdev)
    p, mean, stdev = values_big
    p = np.array(p)
    p = p + 0.005
    plotline_list[1][0].append(p)
    plotline_list[1][1].append(mean)
    plotline_list[1][2].append(stdev)

plt.errorbar(plotline_list[0][0],plotline_list[0][1],plotline_list[0][2], marker="^", color="black", label="smallbox", linestyle = "none")
plt.errorbar(plotline_list[1][0],plotline_list[1][1],plotline_list[1][2], marker = "^", color="orange", label="bigbox", linestyle = "none")
plt.xlabel("Value for P")
plt.ylabel("Found intersecton area")
plt.legend()
plt.show()
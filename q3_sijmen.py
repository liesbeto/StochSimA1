import numpy as np
import matplotlib.pyplot as plt
import statistics as st


def montecarlo_point(upper, lower, P):
    """Creates a random 3D point within a cube with bounds upper and lower."""
    if np.random.random() < P:
        x = np.random.random() * (upper-lower) + lower
        y = np.random.random() * (upper-lower) + lower
        z = np.random.random() * (upper-lower) + lower
        small = False
    else:
        x = np.random.random() * (upper-lower) + lower
        y = np.random.random() * (upper-lower) + lower
        z = np.random.random() * (0.5-(-0.3)) + (-0.3)
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
    intersect_frac_small = intersect_inside_smallbox / N_small
    intersect_size_small = intersect_frac_small * ((0.5-(-0.3)) * (upper - lower) ** 2 )
    intersect_frac_big = intersect_inside_bigbox / N_big
    intersect_size_big = intersect_frac_big * ((upper - lower) ** 3)    
#    visualise_all(list_coords, k)
#    visualise_sphere(list_coords, k)
#    visualise_torus(list_coords, r_maj, r_min, center_coord)
#    visualise_intersection(list_coords, k, r_maj, r_min)
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
    for _ in range(10):
        values = calculate_intersection_boxes(100000, 1.50, -1.15, 1, 0.75, 0.4, center_coord= center_coord, P=p)
        temp_values.append(values)
    values_list.append(((p, st.mean(temp_values[0]), st.stdev(temp_values[0])), (p, st.mean(temp_values[1]), st.stdev(temp_values[1]))))

for values_big, values_small in values_list:
    p, mean, std = values_big
    p = np.array(p)
    p = p - 0.005
    plt.errorbar(p, mean, std, marker = "^", color="black", label = "bigbox")
    p, mean, std = values_small
    p = np.array(p)
    p = p + 0.005
    plt.errorbar(p, mean, std, marker = "^", color="orange", label = "smallbox")

plt.legend()
plt.show()
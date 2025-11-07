import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
np.random.seed(1)

def montecarlo_point(upper, lower, P):
    """Creates a random 3D point within a cube with bounds upper and lower."""
    if np.random.random() < P:
        x = np.random.random() * (upper-lower) + lower
        y = np.random.random() * (upper-lower) + lower
        z = np.random.random() * (upper-lower) + lower
        small = False
    else:
        x = np.random.random() * (1.15-(-1.15)) + (-1.15)
        y = np.random.random() * (1.15-(-1.15)) + (-1.15)
        z = np.random.random() * (0.5-(-0.3)) + (-0.3)
        small = True
    return (x, y, z), small

def logistic_point(coords, m=3.8):
    x, y, z = coords
    x = m*x * (1-x)
    y = m*y * (1-y)
    z = m*z * (1-z)
    coords = x, y, z
    return coords

def in_sphere(coord, k):
    if len(coord) > 2:
        x, y, z = coord
    else:
        x, y, z = coord[0]
    return x**2 + y**2 + z**2 <= k**2

def in_torus(coord, r_maj, r_min, center_coord = [0,0,0]):
    if len(coord) > 2:
        x, y, z = coord
    else:
        x, y, z = coord[0]
    x_c, y_c, z_c = center_coord
    return (np.sqrt((x-x_c)**2 + (y-y_c)**2) - r_maj)**2 + (z-z_c)**2 <= r_min**2

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

def visualise_intersection(list_coords, k, r_maj, r_min, center_coord):
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

def transform_to_bounds(value, upper, lower, mini=0.1805, maxi=0.95):
    return (((value - mini) * (upper - lower)) / (maxi - mini)) + lower

def logistic_sequence(N, upper, lower):
    x, y, z = np.random.random()*(0.95-0.1805)*0.1805, np.random.random()*(0.95-0.1805)*0.1805, np.random.random()*(0.95-0.1805)*0.1805
    lx_logistic, ly_logistic, lz_logistic = [x], [y], [z]
    for _ in range(N):
        x, y, z = logistic_point((x,y,z))
        lx_logistic.append(x)
        ly_logistic.append(y)
        lz_logistic.append(z)

    list_coords = []
    for i in range(len(lx_logistic)):
        list_coords.append((transform_to_bounds(lx_logistic[i], upper, lower), transform_to_bounds(ly_logistic[i], upper, lower), transform_to_bounds(lz_logistic[i], upper, lower)))

    return list_coords

def logistic_intersection(N, upper, lower, k, r_maj, r_min, visualise=False):
    list_intersect = []

    list_coords = logistic_sequence(N, upper, lower)
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

def montecarlo_sequence(N, upper, lower, P=1):
    list_coords = []
    for _ in range(N):
        coords = montecarlo_point(upper, lower, P)
        list_coords.append(coords)
    
    return list_coords

def montecarlo_intersection(N, upper, lower, k, r_maj, r_min, visualise=False, P = 1):
    list_intersect = []
    
    list_coords = montecarlo_sequence(N, upper, lower)
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

def plot_freqdist(N, upper, lower, func=montecarlo_sequence):
    lx = []

    list_coords = func(N, upper, lower)
    for coord in list_coords:
        x = list(coord)[0][0]
        lx.append(x)

    if func == montecarlo_sequence:
        title = "Uniform Random Samples Distribution"
    else:
        title = "Logistic Sequence Samples Distribution"

    plt.hist(lx, bins=100)
    plt.xlabel("Sample Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

def question2_plot(case, bounds):
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
            uni_temp.append(montecarlo_intersection(100000, bound, -1*bound, k, r_maj, r_min))
            log_temp.append(logistic_intersection(100000, bound, -1*bound, k, r_maj, r_min))
        uniform_randoms.append(np.mean(uni_temp))
        uniform_randoms_err.append(np.std(uni_temp))
        logistics.append(np.mean(log_temp))
        logistics_err.append(np.std(log_temp))

    plt.errorbar(bounds, uniform_randoms, uniform_randoms_err, label="Uniform")
    plt.errorbar(bounds, logistics, logistics_err, label="Logistic")
    plt.legend()
    plt.show()

def calculate_intersection_boxes(N, upper, lower, k, r_maj, r_min, sampling_method="montecarlo", coords_0 = None, center_coord=[0,0,0], P=1, return_coords=False):
    intersect_inside_bigbox = 0
    intersect_inside_smallbox = 0
    N_small=0
    N_big=0
    list_coords = []
    just_started = True
    for _ in range(N):
        if sampling_method == "montecarlo":
            coords, small = montecarlo_point(upper, lower, P)
        else:
            if just_started:
                coords = logistic_point(coords_0)
            else:
                coords = logistic_point(list_coords[-1])
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



    if N_small > 0:
        intersect_frac_small = intersect_inside_smallbox / N_small
        intersect_size_small = intersect_frac_small * ((0.5-(-0.3)) * (1.15 - (-1.15)) ** 2 )
    if N_big > 0:
        intersect_frac_big = intersect_inside_bigbox / N_big
        intersect_size_big = intersect_frac_big * ((upper - lower) ** 3)    

    if return_coords:
        hit_coords = [coord for coord in list_coords if in_sphere(coord, k) and in_torus(coord, r_maj, r_min, center_coord)]
        if N_small > 0 and N_big == 0:
            return intersect_size_small, hit_coords
        if N_big > 0 and N_small == 0:
            return intersect_size_big, hit_coords
        else:
            return (intersect_size_small, intersect_size_big), hit_coords
    else:
        if N_small > 0 and N_big == 0:
            return intersect_size_small
        if N_big > 0 and N_small == 0:
            return intersect_size_big
        else:
            return intersect_size_small, intersect_size_big


def question3_plot(repeats = 100, center_coord=[0, 0, 0.1]):
    values_list = []
    for prob in range(1,10):
        p = prob/10
        temp_values = []
        for _ in range(repeats):
            values = calculate_intersection_boxes(100000, 1.2, -1.2, 1, 0.75, 0.4, center_coord= center_coord, P=p)
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
    return

def ex1(visualize=False):
	print("Ex1 - sphere intersect torus, uniform:")
	
	b_box = np.array([[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]])
	b_box_size = np.array([2.3, 2.3, 2.3])

	cases = [(1, 0.75, 0.4),
			 (1, 0.5 , 0.5)]
	
	data = []
	for i, params in enumerate(cases):
		r, t_R, t_r = params
		volume, hits = calculate_intersection_boxes(100000, 1.15, -1.15, r, t_R, t_r, center_coord=[0,0,0], P=1, return_coords=True)
		hits = np.array(hits)
		data.append((volume, hits))
		print(f"  case {i+1}: {volume:.5f}")

	if visualize:
		fig = plt.figure()

		for i, ((volume, hits), params) in enumerate(zip(data, cases)):
			r, t_R, t_r = params

            # 3d scatter plot of samples
			ax = fig.add_subplot(2, 2, (i+1), projection="3d")
			ax.set_title(f"case {i+1}: $r_{{sphere}}$ = {r}, $R_{{torus}}$ = {t_R}, $r_{{torus}}$ = {t_r}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.set_zlabel("z")
			ax.set_box_aspect([1, 1, 1])
			ax.view_init(elev=40, azim=-60)

			xs = hits[:,0]
			ys = hits[:,1]
			zs = hits[:,2]

			ax.scatter(xs, ys, zs, s=1, alpha=0.1)

            # 2d surface along the x-axis
			x = np.linspace(b_box[0,0], b_box[0,1], 2)
			z = np.linspace(b_box[2,0], b_box[2,1], 2)
			X, Z = np.meshgrid(x, z)
			Y = np.full_like(X, 0)
			ax.plot_surface(X, Y, Z, color='grey', alpha=0.2)

			torus1 = plt.Circle(( t_R, 0), t_r, color='r', fill=False, label="torus")
			torus2 = plt.Circle((-t_R, 0), t_r, color='r', fill=False, label="torus")
			sphere = plt.Circle((   0, 0), r  , color='g', fill=False, label="sphere")
			b_box_rect = plt.Rectangle((b_box[0,0], b_box[2,0]), b_box_size[0], b_box_size[2], color='b', fill=False, label="bounding box")
			ax.add_patch(torus1)
			ax.add_patch(torus2)
			ax.add_patch(sphere)
			ax.add_patch(b_box_rect)
			art3d.pathpatch_2d_to_3d(torus1, z=0, zdir="y")
			art3d.pathpatch_2d_to_3d(torus2, z=0, zdir="y")
			art3d.pathpatch_2d_to_3d(sphere, z=0, zdir="y")
			art3d.pathpatch_2d_to_3d(b_box_rect, z=0, zdir="y")

			ax.set_xlim(b_box[0,0], b_box[0,1])
			ax.set_ylim(b_box[1,0], b_box[1,1])
			ax.set_zlim(b_box[2,0], b_box[2,1])

            # 2d slice along x-axis: analytical circles
			ax = fig.add_subplot(2, 2, (i+1)+2)
			ax.set_title("2d slice along x-axis")
			ax.axis('equal')
			ax.set_xlabel("x")
			ax.set_ylabel("z")
			torus1 = plt.Circle(( t_R, 0), t_r, color='r', fill=False, label="torus")
			torus2 = plt.Circle((-t_R, 0), t_r, color='r', fill=False, label="torus")
			sphere = plt.Circle((   0, 0), r  , color='g', fill=False, label="sphere")
			b_box_rect = plt.Rectangle((b_box[0,0], b_box[2,0]), b_box_size[0], b_box_size[2], color='b', fill=False, label="bounding box")
			ax.add_patch(torus1)
			ax.add_patch(torus2)
			ax.add_patch(sphere)
			ax.add_patch(b_box_rect)

			mask_slice = (hits[:,1] > -0.05) & (hits[:,1] < 0.05)
			x = hits[mask_slice,0]
			y = hits[mask_slice,2]
			ax.scatter(x, y, s = 1, label="samples")

		plt.tight_layout()

		handles, labels = plt.gca().get_legend_handles_labels()
		dict_of_labels = dict(zip(labels, handles))
		plt.legend(dict_of_labels.values(), dict_of_labels.keys())
		plt.show()

def main():
    ex1(visualize=True)
    plot_freqdist(100000, 1.15, -1.15, montecarlo_sequence)
    question2_plot("A", [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3])
    question3_plot(repeats=100)
    
if __name__ == "__main__":
    main()
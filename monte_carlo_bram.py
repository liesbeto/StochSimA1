import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d


# --- Parameters ---
n_samples = 100000
b_box = np.array([[-1.4, 1.4], [-1.4, 1.4], [-1.2, 1.2]])
b_box_size = b_box[:,1] - b_box[:, 0]
b_box_volume = np.prod(b_box_size)
print("bounding box volume:", b_box_volume)


# --- Shapes ---
def in_sphere(x, y, z, r): return x**2 + y**2 + z**2 <= r**2
def in_torus(x, y, z, R, r): return ((x**2 + y**2)**0.5 - R)**2 + z**2 <= r**2


# --- Samplers ---
def uniform_sampler(b_box):
	return np.random.uniform(low=b_box[:,0], high=b_box[:,1])

log_map_sample = [0.1, 0.2, 0.3]
def logistic_map_sampler(b_box):
	global log_map_sample
	m = 3.8
	log_map_sample = [m * x * (1 - x) for x in log_map_sample]

	b_box_origin = b_box[:,0]
	b_box_size = b_box[:,1] - b_box[:, 0]
	return b_box_origin + log_map_sample * b_box_size

def mixture_sampler(b_box):
	if np.random.rand() < 0.8:
		return uniform_sampler(b_box)
	small_box = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.4, 0.6]])
	return uniform_sampler(small_box)


# --- Monte Carlo ---
def monte_carlo_3d(b_box, in_shape_func, sampler):
	hits = []
	n_hit = 0
	for _ in range(n_samples):
		sample = sampler(b_box)
		x, y, z = sample

		if in_shape_func(x, y, z):
			n_hit += 1
			hits.append(sample)

	hits = np.array(hits)
	volume = n_hit / n_samples * b_box_volume
	return volume, hits


# --- Excercises ---
def ex1(visualize=False):
	print("Ex1 - sphere intersect torus, uniform:")

	cases = [(1, 0.75, 0.4), # case a
			 (1, 0.5 , 0.5)] # case b
	
	data = []
	for i, params in enumerate(cases):
		r, t_R, t_r = params
		f = lambda x, y, z: in_sphere(x, y, z, r) and in_torus(x, y, z, t_R, t_r) # case a
		volume, hits = monte_carlo_3d(b_box, f, uniform_sampler)
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

			# 2d slice along x-axis: samples
			mask_slice = (hits[:,1] > -0.05) & (hits[:,1] < 0.05)
			x = hits[mask_slice,0]
			y = hits[mask_slice,2]
			ax.scatter(x, y, s = 1, label="samples")

		plt.tight_layout()
		plt.suptitle(f"\
Monte carlo integration: intersection of sphere and torus \n\
bounding box: \n\
$x_{{min}}$ = {b_box[0,0]}, $x_{{max}}$ = {b_box[0,1]}\n\
$y_{{min}}$ = {b_box[0,0]}, $y_{{max}}$ = {b_box[0,1]}\n\
$z_{{min}}$ = {b_box[0,0]}, $z_{{max}}$ = {b_box[0,1]}")

		# remove doulbe labels
		handles, labels = plt.gca().get_legend_handles_labels()
		dict_of_labels = dict(zip(labels, handles))
		plt.legend(dict_of_labels.values(), dict_of_labels.keys())

		plt.show()

def ex2():
	print("Ex2 - logistic sampling:")
	f = lambda x, y, z: in_sphere(x, y, z, 1) and in_torus(x, y, z, 0.75, 0.4)
	volume, _ = monte_carlo_3d(b_box, f, logistic_map_sampler)
	print(f"  V = {volume:0.5f}")

def ex3(visualize=False):
	print("Ex3 - mixture sampling:")
	f = lambda x, y, z: in_sphere(x, y, z, 1) and in_torus(x, y, z - 0.1, 0.75, 0.4)
	volume_uniform, _    = monte_carlo_3d(b_box, f, uniform_sampler)
	volume_mixture, hits = monte_carlo_3d(b_box, f, mixture_sampler)
	print(f"  uniform: {volume_uniform:.5f}")
	print(f"  mixture: {volume_mixture:.5f}")

	if visualize:
		ax = plt.figure().add_subplot(111, projection="3d")

		xs = hits[:,0]
		ys = hits[:,1]
		zs = hits[:,2]

		ax.scatter(xs, ys, zs, s = 1)
		ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

		plt.show()

ex1(visualize=True)
# ex2()
# ex3()


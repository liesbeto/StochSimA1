import matplotlib.pyplot as plt
import numpy as np


# --- Parameters ---
n_samples = 100000
b_box = np.array([[-2, 2], [-2, 2], [-2, 2]])
b_box_volume = np.prod(b_box[:,1] - b_box[:, 0])
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
def ex1():
	print("Ex1 - sphere intersect torus, uniform:")

	f1 = lambda x, y, z: in_sphere(x, y, z, 1) and in_torus(x, y, z, 0.75, 0.4) # case a
	f2 = lambda x, y, z: in_sphere(x, y, z, 1) and in_torus(x, y, z, 0.5, 0.5)# case b
	
	for i, f in enumerate([f1, f2], 1): # enumerate starting from 1 instead of 0
		volume, _ = monte_carlo_3d(b_box, f, uniform_sampler)
		print(f"  case {i}: {volume:.5f}")

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

ex1()
ex2()
ex3(visualize=True)


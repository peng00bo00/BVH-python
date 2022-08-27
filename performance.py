from typing import Tuple
import igl
import numpy as np
import matplotlib.pyplot as plt

import time
from functools import wraps

from BVH import Ray, BVH, Triangle, buildBVH


## timeit decorator
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        total_time = end - start

        return total_time, result
    
    return timeit_wrapper

## time profile
@timeit
def brute_force_loop(rays: list[Ray], triangles: list[Triangle]) -> Tuple[list[bool], list[float], list[int]]:
    """Brute-force loop.

    Args:
        rays: a list of rays
        triangles: a list of triangles
    
    Return:
        hits: whether the ray hits a triangle
        ts: time of fligts
        indices: list of hitted triangle index
    """

    hits, ts, indices = [], [], []
    for ray in rays:
        for triangle in triangles:
            hit, t = triangle.intersect(ray)

            if hit:
                hits.append(hit)
                ts.append(t)
                indices.append(triangle.faceIdx)

                break
    
    return hits, ts, indices

@timeit
def bvh_acceleration(rays: list[Ray], bvh: BVH) -> Tuple[list[bool], list[float], list[int]]:
    """Use BVH for acceleration.

    Args:
        rays: a list of rays
        bvh: a BVH instance
    
    Return:
        hits: whether the ray hits a triangle
        ts: time of fligts
        indices: list of hitted triangle index
    """

    hits, ts, indices = [], [], []
    for ray in rays:
        hit, t, idx = bvh.intersect(ray)

        hits.append(hit)
        ts.append(t)
        indices.append(bvh.triangles[idx].faceIdx)
    
    return hits, ts, indices


## initialize random rays
NUM_RAYS = 1000
rays = [Ray(np.zeros(3), np.random.random(3)) for _ in range(NUM_RAYS)]

## test on different meshes
Fs = []
Ts = []
Acc= []

for i in range(5):
    ## initialize triangles and BVH
    V, F = igl.read_triangle_mesh(f"./mesh/standard_sphere_{i}.obj")

    triangles = [Triangle(V[F[i]], i) for i in range(len(F))]
    bvh = buildBVH(V, F)

    ## brute force loop
    tc1, (_, t1, F1) = brute_force_loop(rays, triangles)

    ## BVH
    tc2, (_, t2, F2) = bvh_acceleration(rays, bvh)

    assert np.allclose(t1, t2)
    assert np.array_equal(F1, F2)

    acc = tc1 / tc2

    print(f"Test on {len(F)} faces with {NUM_RAYS} random rays:")
    print(f"\tBrute-Force loop takes {tc1:.3f} s")
    print(f"\tBVH takes {tc2:.3f} s")
    print(f"\t{acc:.3f}x speed up!")
    print()

    Fs.append(len(F))
    Ts.append((tc1, tc2))
    Acc.append(acc)

## plots
Ts = np.array(Ts)

plt.plot(Fs, Ts[:, 0], label="Brute-Force Loop")
plt.plot(Fs, Ts[:, 1], label="BVH")
plt.xlabel("Num. of Faces")
plt.ylabel("Time Cost (s)")
plt.yscale('log')
plt.title(f"Time Cost on {NUM_RAYS} Random Rays")
plt.legend()
plt.savefig("./plot/time_cost.png", dpi=300, bbox_inches='tight')
plt.show()

plt.plot(Fs, Acc)
plt.xlabel("Num. of Faces")
plt.ylabel("Speed-Up (x)")
plt.title(f"Speed-Up on {NUM_RAYS} Random Rays")
plt.savefig("./plot/speed_up.png", dpi=300, bbox_inches='tight')
plt.show()
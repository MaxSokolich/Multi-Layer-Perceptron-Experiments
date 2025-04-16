import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
import numpy as np


def greedy_min_deviation_ordering(velocities, durations):
    N = len(velocities)
    d = len(velocities[0])

    # Compute the total displacement vector
    s_list = [v * t for v, t in zip(velocities, durations)]
    total_disp = sum(s_list)

    # Normalize direction of total displacement
    total_dir = total_disp / np.linalg.norm(total_disp)

    # Build orthogonal projection matrix to total direction
    P = np.eye(d) - np.outer(total_dir, total_dir)

    remaining = list(range(N))
    ordering = []
    current_dev = np.zeros(d)

    for _ in range(N):
        best_idx = None
        best_dev = None

        for i in remaining:
            step = s_list[i]
            step_perp = P @ step
            new_dev = current_dev + step_perp
            norm_dev = np.linalg.norm(new_dev)

            if best_dev is None or norm_dev < best_dev:
                best_dev = norm_dev
                best_idx = i

        ordering.append(best_idx)
        current_dev += P @ s_list[best_idx]
        remaining.remove(best_idx)

    return ordering


def find_max_scale_to_fit_box(polygon, box_bounds = [(-90,90),(-90,90),(-90,90),(-90,90)]):
    """
    Parameters:
    - polygon: List of N-dimensional points (tuples or lists).
    - fixed_corner: A single N-dimensional point.
    - box_bounds: A list of (min_i, max_i) tuples for each dimension.
    
    Returns:
    - max_scale: The largest possible scale factor (<= 1.0).
    """


    polygon = np.array(polygon)
    fixed_corner = polygon[0]
    fixed_corner = np.array(fixed_corner)
    dims = len(fixed_corner)
    max_scale = 1.0

    for point in polygon:
        if np.allclose(point, fixed_corner):
            continue  # Skip the fixed point

        direction = point - fixed_corner
        for i in range(dims):
            if direction[i] == 0:
                continue

            s_min = (box_bounds[i][0] - fixed_corner[i]) / direction[i]
            s_max = (box_bounds[i][1] - fixed_corner[i]) / direction[i]

            # Keep valid s within (0,1]
            valid_scales = []
            if direction[i] > 0:
                if 0 < s_max <= 1:
                    valid_scales.append(s_max)
            else:
                if 0 < s_min <= 1:
                    valid_scales.append(s_min)

            if valid_scales:
                max_scale = min(max_scale, min(valid_scales))

    return max_scale

def compute_min_n_by_polygon_shrinkage(velocities, durations, x0, box_lower, box_upper):
    N = len(velocities)
    d = len(x0)

    # Step 1: Compute the full polygonal path
    p = [x0.copy()]
    for i in range(N):
        p.append(p[-1] + velocities[i] * durations[i])

    S = p[-1] - x0  # total displacement
    shape_vectors = [pi - x0 for pi in p]  # relative polygon shape

    # Step 2: For each dimension, find how much room we have along the segment
    tightest_n = 0
    for j in range(d):
        # Compute min margin along the segment in dim j
        x_j_min = min(x0[j], x0[j] + S[j])
        x_j_max = max(x0[j], x0[j] + S[j])
        margin_j = min(box_upper[j] - x_j_max, x_j_min - box_lower[j])

        if margin_j <= 0:
            return None  # even the line doesn't fit

        # Max absolute deviation of polygon in this dimension
        max_dev_j = max(abs(s[j]) for s in shape_vectors)

        # Required n in this dimension
        n_j = max_dev_j / margin_j
        tightest_n = max(tightest_n, n_j)

    return int(np.ceil(tightest_n))

def compute_exact_min_n(velocities, durations, x0, box_lower, box_upper):
    """
    velocities: list of np.arrays of shape (d,)
    durations: list of scalars of same length
    x0: np.array of shape (d,)
    box_lower: np.array of shape (d,)
    box_upper: np.array of shape (d,)
    """

    N = len(velocities)
    d = len(x0)

    # Step 1: Compute displacement vectors and total displacement
    s = [v * t for v, t in zip(velocities, durations)]
    S = sum(s)  # total displacement

    # Step 2: Build the deviation set Gamma
    gamma = []
    p = np.zeros(d)
    for k in range(N + 1):
        alpha = k / N
        target = alpha * S
        deviation = p - target
        gamma.append(deviation.copy())
        if k < N:
            p += s[k]

    # Step 3: For each gamma and each dimension, compute required n
    max_n = 0
    for g in gamma:
        for j in range(d):
            # Consider the path along the line x0 + alpha * S
            x_line_min = min(x0[j], x0[j] + S[j])
            x_line_max = max(x0[j], x0[j] + S[j])

            margin_upper = box_upper[j] - x_line_max
            margin_lower = x_line_min - box_lower[j]

            if g[j] > 0:
                if margin_upper <= 0:
                    return None  # violates box no matter what
                n_j = g[j] / margin_upper
            elif g[j] < 0:
                if margin_lower <= 0:
                    return None
                n_j = -g[j] / margin_lower
            else:
                n_j = 0

            max_n = max(max_n, n_j)


    
    return int(np.ceil(max_n))
def find_max_scale_to_fit_box(polygon, box_bounds = [(-90,90),(-90,90),(-90,90),(-90,90)]):
    """
    Parameters:
    - polygon: List of N-dimensional points (tuples or lists).
    - fixed_corner: A single N-dimensional point.
    - box_bounds: A list of (min_i, max_i) tuples for each dimension.
    
    Returns:
    - max_scale: The largest possible scale factor (<= 1.0).
    """


    polygon = np.array(polygon)
    fixed_corner = polygon[0]
    fixed_corner = np.array(fixed_corner)
    dims = len(fixed_corner)
    max_scale = 1.0

    for point in polygon:
        if np.allclose(point, fixed_corner):
            continue  # Skip the fixed point

        direction = point - fixed_corner
        for i in range(dims):
            if direction[i] == 0:
                continue

            s_min = (box_bounds[i][0] - fixed_corner[i]) / direction[i]
            s_max = (box_bounds[i][1] - fixed_corner[i]) / direction[i]

            # Keep valid s within (0,1]
            valid_scales = []
            if direction[i] > 0:
                if 0 < s_max <= 1:
                    valid_scales.append(s_max)
            else:
                if 0 < s_min <= 1:
                    valid_scales.append(s_min)

            if valid_scales:
                max_scale = min(max_scale, min(valid_scales))

    return max_scale

class Chop:
    def __init__(self, velocities,durations, x0):
        """
        polygon: list of points in R^d, where each point is a list or np.array of length d
        """
        # self.polygon = [np.array(p) for p in polygon]
        self.velocities = velocities
        self.durations = durations
        self.x0 = x0
        self.ordering = greedy_min_deviation_ordering(velocities, durations)
        self.velocities = [velocities[i] for i in self.ordering]
        self.durations = [durations[i] for i in self.ordering]

    def get_chopped_trajectory(self, n):
        """
        n: number of chopped points
        """
        # n = self.get_chop_step()
        chopped_points = [self.x0]
        for j in range(n):
            for i in range(len(self.velocities)):
                chopped_points.append(chopped_points[-1] + self.velocities[i]*self.durations[i]*((1)/n))
        return np.array(chopped_points)


    
    def visualize(self):
        polygon_np = np.array(self.polygon)
        farthest_point, dist = self.find_farthest_corner()

        plt.figure(figsize=(8, 6))
        plt.plot(polygon_np[:, 0], polygon_np[:, 1], 'bo-', label='Polygon')
        plt.plot(
            [self.polygon[0][0], self.polygon[-1][0]],
            [self.polygon[0][1], self.polygon[-1][1]],
            'r--', label='Line (First to Last Point)'
        )
        plt.plot(farthest_point[0], farthest_point[1], 'ro', markersize=10, label='Farthest Corner')
        plt.title("Farthest Corner from Line Segment (First to Last Point)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

   

    def get_chop_step(self, box_min = np.array([-100]*4), box_max = np.array([100]*4)):
        ## find the max scale to fit the box
        # print("Finding max scale to fit boxxx")
        n = compute_min_n_by_polygon_shrinkage(self.velocities, self.durations, self.x0, box_min, box_max)
        if n is not None:
            return n
        else:
            n = int(1/find_max_scale_to_fit_box( self.get_chopped_trajectory(1)))+1
            while True:
                trj = self.get_chopped_trajectory(n)
                outbound = np.max(np.abs(trj))
                if (outbound < box_max).all() and (outbound > box_min).all():
                    break
                else:
                    n = int(n*np.max(outbound/box_max))+1
                    print(n)
        return n
# Example polygon and visualization
polygon = [
    [0, 0],
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 0]
]

# chop = Chop(polygon)
# farthest_point, distance = chop.find_farthest_corner()
# print("Farthest Point:", farthest_point)
# print("Distance:", distance)
# chop.visualize()






# # Example
# p0 = np.array([0.5, 0.2])
# p1 = np.array([1.0, 1.0])
# box_min = np.array([-1.0, -1.0])
# box_max = np.array([2.0, 2.0])

# dist, corner = max_distance_to_line_from_box(p0, p1, box_min, box_max, visualize=True)
# print("Max normal distance:", dist)
# print("Farthest corner:", corner)



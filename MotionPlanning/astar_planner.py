import numpy as np
import matplotlib.pyplot as plt
import heapq

# -------------------------------------------------------------
# 1. 2D Grid A* Implementation (Part B d)
# -------------------------------------------------------------
class Node:
    def __init__(self, x, y, cost, parent_idx):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_idx = parent_idx

    def __lt__(self, other):
        return self.cost < other.cost

def a_star_planning(ox, oy, reso, start, goal, grid_bounds):
    """
    ox, oy: Obstacle x and y positions
    reso: Grid resolution in meters
    start: [sx, sy]
    goal: [gx, gy]
    grid_bounds: [min_x, max_x, min_y, max_y]
    """
    min_x, max_x, min_y, max_y = grid_bounds
    x_width = int(round((max_x - min_x) / reso))
    y_width = int(round((max_y - min_y) / reso))

    # Obstacle map
    obmap = np.zeros((x_width, y_width), dtype=bool)
    for ix, iy in zip(ox, oy):
        idx_x = int(round((ix - min_x) / reso))
        idx_y = int(round((iy - min_y) / reso))
        if 0 <= idx_x < x_width and 0 <= idx_y < y_width:
            obmap[idx_x, idx_y] = True

    # 8-connected motion primitives: [dx, dy, cost]
    motion = [
        [1, 0, 1.0], [0, 1, 1.0], [-1, 0, 1.0], [0, -1, 1.0],
        [1, 1, np.sqrt(2)], [1, -1, np.sqrt(2)], [-1, 1, np.sqrt(2)], [-1, -1, np.sqrt(2)]
    ]

    sx_idx = int(round((start[0] - min_x) / reso))
    sy_idx = int(round((start[1] - min_y) / reso))
    gx_idx = int(round((goal[0] - min_x) / reso))
    gy_idx = int(round((goal[1] - min_y) / reso))

    start_node = Node(sx_idx, sy_idx, 0.0, -1)
    
    open_set = []
    # priority: cost + heuristic
    heapq.heappush(open_set, (np.hypot(sx_idx - gx_idx, sy_idx - gy_idx), 0, start_node))
    closed_set = {}
    explored_nodes = []

    node_count = 0
    while open_set:
        _, current_id, current = heapq.heappop(open_set)
        curr_key = (current.x, current.y)

        if curr_key in closed_set:
            continue
        closed_set[curr_key] = current
        explored_nodes.append((current.x * reso + min_x, current.y * reso + min_y))

        # Check goal reached
        if current.x == gx_idx and current.y == gy_idx:
            break

        for dx, dy, move_cost in motion:
            nx = current.x + dx
            ny = current.y + dy
            if not (0 <= nx < x_width and 0 <= ny < y_width) or obmap[nx, ny]:
                continue
            if (nx, ny) in closed_set:
                continue

            g_cost = current.cost + move_cost
            h_cost = np.hypot(nx - gx_idx, ny - gy_idx)
            new_node = Node(nx, ny, g_cost, curr_key)
            node_count += 1
            heapq.heappush(open_set, (g_cost + h_cost, node_count, new_node))

    # Reconstruct path
    rx, ry = [], []
    curr = closed_set.get((gx_idx, gy_idx))
    while curr and curr.parent_idx != -1:
        rx.append(curr.x * reso + min_x)
        ry.append(curr.y * reso + min_y)
        curr = closed_set.get(curr.parent_idx)
    rx.append(start[0])
    ry.append(start[1])

    return rx[::-1], ry[::-1], explored_nodes

# -------------------------------------------------------------
# 2. Curvature & Radius Verification (Part B f)
# -------------------------------------------------------------
def check_path_feasibility(rx, ry, r_min=3.68):
    """
    Computes local turning radius via circumscribed circle around 3 consecutive points.
    Flags points violating R >= R_min.
    """
    violations = []
    radii = []
    
    for i in range(1, len(rx) - 1):
        p1 = np.array([rx[i-1], ry[i-1]])
        p2 = np.array([rx[i], ry[i]])
        p3 = np.array([rx[i+1], ry[i+1]])

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Cross product to check area
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
        if area > 1e-4:
            radius = (a * b * c) / (4.0 * area)
        else:
            radius = np.inf

        radii.append(radius)
        if radius < r_min:
            violations.append((rx[i], ry[i], radius))

    return radii, violations

# -------------------------------------------------------------
# 3. Environment Setup & Execution (Part B e)
# -------------------------------------------------------------
reso = 1.0  # 1-meter grid resolution
bounds = [0, 40, 0, 30]

# Build a road with a bend and obstacles
ox, oy = [], []
# Bottom wall
for x in range(0, 40):
    ox.append(x); oy.append(0)
    ox.append(x); oy.append(30)
for y in range(0, 30):
    ox.append(0); oy.append(y)
    ox.append(40); oy.append(y)

# Static obstacles (simulating campus road layout / barriers)
obstacles = [
    (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), # Choke barrier
    (20, 18), (21, 18), (22, 18), (23, 18),           # Parked buggy obstacle
    (28, 8), (28, 9), (28, 10), (28, 11), (28, 12),  # Curve obstacle
    (15, 24), (16, 24), (17, 24)                      # Road debris
]
for obs in obstacles:
    ox.append(obs[0]); oy.append(obs[1])

start = [4.0, 5.0]
goal = [35.0, 25.0]

rx, ry, explored = a_star_planning(ox, oy, reso, start, goal, bounds)
radii, violations = check_path_feasibility(rx, ry, r_min=3.68)

# -------------------------------------------------------------
# 4. Visualization
# -------------------------------------------------------------
plt.figure(figsize=(10, 7))
exp_x, exp_y = zip(*explored)
plt.scatter(exp_x, exp_y, c='#e2e8f0', s=15, label='Explored Nodes (A*)')
plt.scatter(ox, oy, c='black', marker='s', s=45, label='Static Obstacles')
plt.plot(rx, ry, 'b-o', linewidth=2, markersize=4, label='A* Planned Path')

if violations:
    vx, vy, _ = zip(*violations)
    plt.scatter(vx, vy, c='red', s=120, edgecolors='black', zorder=5, 
                label=f'Infeasible Turns (R < 3.68m) [{len(violations)} pts]')

plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
plt.plot(goal[0], goal[1], 'mo', markersize=10, label='Goal')
plt.title("Grid A* vs Buggy Minimum Turning Constraint (R_min = 3.68 m)")
plt.xlabel("X [m]"); plt.ylabel("Y [m]")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Total Path Waypoints: {len(rx)}")
print(f"Total Infeasible Points Detected: {len(violations)}")
import numpy as np
import matplotlib.pyplot as plt

# 1. SIMULATOR & EGO POLICY (Part D h)
def run_sim(d_init, v_adv_x, v_adv_y):
    dt, max_t = 0.1, 4.0
    x_ego, v_ego = 0.0, 5.0          # Ego starts at 0 m, cruising at 5 m/s (~18 km/h)
    x_adv, y_adv = d_init, 3.0       # Adversary starts ahead, 3 m to the side
    
    ttc_list, dist_list = [], []
    ego_x_hist, adv_x_hist, adv_y_hist = [], [], []

    for _ in range(int(max_t / dt)):
        # Rule-based ego policy: brake if adversary is within 10 m and enters lane (|y| < 1.5 m)
        a_ego = -3.0 if (0 < (x_adv - x_ego) < 10.0 and abs(y_adv) < 1.5) else 0.0
        
        # Step positions
        v_ego = max(0.0, v_ego + a_ego * dt)
        x_ego += v_ego * dt
        x_adv += v_adv_x * dt
        y_adv += v_adv_y * dt
        
        # Record data
        dx = x_adv - x_ego
        dist = np.hypot(dx, y_adv)
        rel_v = v_ego - v_adv_x
        ttc = (dx / rel_v) if (rel_v > 0 and dx > 0) else 99.0
        
        ttc_list.append(ttc)
        dist_list.append(dist)
        ego_x_hist.append(x_ego)
        adv_x_hist.append(x_adv)
        adv_y_hist.append(y_adv)
        
        if dist < 1.5:  # Collision threshold
            break
            
    return min(ttc_list), min(dist_list), ego_x_hist, adv_x_hist, adv_y_hist

# 2. RANDOM SEARCH OVER 3 PARAMETERS (Part D i)
np.random.seed(42)
scenarios = []

for i in range(30):
    d_init = np.random.uniform(6.0, 18.0)    # initial gap (m)
    v_adv_x = np.random.uniform(1.0, 3.0)   # forward speed (m/s)
    v_adv_y = np.random.uniform(-1.8, -0.6) # crossing speed (m/s)
    
    min_ttc, min_dist, ex, ax, ay = run_sim(d_init, v_adv_x, v_adv_y)
    
    # Flag dangerous cases: TTC < 1.5 s or distance < 2.0 m
    if min_ttc < 1.5 or min_dist < 2.0:
        scenarios.append((i, d_init, v_adv_x, v_adv_y, min_ttc, min_dist, ex, ax, ay))

print(f"Tested 30 runs -> Found {len(scenarios)} dangerous scenarios.")
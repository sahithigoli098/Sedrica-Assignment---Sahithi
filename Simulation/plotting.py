# 3. PLOT DISCOVERED CASES (Part D j)
plt.figure(figsize=(10, 4))
for idx in range(min(2, len(scenarios))):
    sc = scenarios[idx]
    plt.subplot(1, 2, idx + 1)
    plt.plot(sc[6], [0]*len(sc[6]), 'b-', label='Ego (in lane)')
    plt.plot(sc[7], sc[8], 'r--', label='Adversary')
    plt.axhline(1.5, color='gray', linestyle=':'); plt.axhline(-1.5, color='gray', linestyle=':')
    plt.title(f"Scenario #{sc[0]}: Min TTC = {sc[4]:.2f}s, Min Dist = {sc[5]:.2f}m")
    plt.xlabel("X Position (m)"); plt.ylabel("Y Position (m)")
    plt.legend()
plt.tight_layout()
plt.show()
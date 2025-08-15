import matplotlib.pyplot as plt
import numpy as np

# Data
agents = np.array([1, 10, 20, 30, 40, 50])
time_s = np.array([39690, 14050, 7428, 6895, 6280, 5728])

# --- Ideal strong-scaling curve: T1 / N ---
k = time_s[0]  # since N=1 => T1 = k
p = np.log(time_s[0] / time_s[-1]) / np.log(agents[-1] / agents[0])
p=1
ideal_times = k / (agents ** p)


# Plot
plt.figure(figsize=(10, 5))
plt.yscale('log')
bar_width = 2
plt.bar(agents, time_s, width=bar_width, color='#3A86FF', label="Agentic System")

# Ideal dashed line
plt.plot(agents, ideal_times, '--', color='black', linewidth=4, label='Ideal')

# Aesthetics
plt.xlabel('#Agent', fontsize=24)
plt.ylabel('Time (s)', fontsize=24)
plt.ylim(1e2, 1e5)
plt.legend(fontsize=24)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("/lustre/orion/stf218/proj-shared/brave/vllm_test/plots/strong.png")
# plt.show()

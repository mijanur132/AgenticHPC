import matplotlib.pyplot as plt
import numpy as np

# Data
agents = [1,  10, 20, 30, 40, 50]
time_s = [405,  650, 791, 985, 1256, 1510]

# Ideal curve with exponent p = 0.3
p = 0.3
ideal_times = [time_s[0] * (a / agents[0])**p for a in agents]

# Adjusted times (unchanged in this case)
adjusted_time_s = time_s.copy()

# Plot
plt.figure(figsize=(10, 5))

# Bar plot for actual times
bar_width = 2
plt.bar(agents, adjusted_time_s, width=bar_width, color='#3A86FF', label="AgentX")

# Ideal (power-law) scaling line
# plt.plot(agents, ideal_times, '--', color='black', linewidth=2, label='Ideal (p=0.3)')

# Perfect weak scaling (flat line)
plt.hlines(time_s[0], xmin=min(agents), xmax=max(agents),
           colors='black', linestyles='dashed', linewidth=4, label='Ideal')


# Aesthetics
plt.yscale('log')
plt.xlabel('#Agent', fontsize=24)
plt.ylabel('Time (s)', fontsize=24)
plt.legend(fontsize=24)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.ylim(1e2, 1e4)
plt.tick_params(axis='both', labelsize=18)
plt.tight_layout()


# Aesthetics
# plt.yscale('log')
# plt.xlabel('#Agent', fontsize = 24)
# plt.ylabel('Time (s)', fontsize = 24)
# #plt.title('Scaling behavior of AgentX', fontsize = 24)
# plt.legend(fontsize=24)
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.ylim(1e2, 1e4)
# plt.tick_params(axis='both', labelsize=18)
# plt.tight_layout()

plt.savefig("/lustre/orion/stf218/proj-shared/brave/vllm_test/plots/weak.png")

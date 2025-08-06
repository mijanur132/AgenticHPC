import matplotlib.pyplot as plt
import numpy as np

# Loss data
start_loss = [
    1.433259, 1.344193, 1.259787, 1.191402, 1.133996, 1.086887, 1.048199, 1.015352,
    0.98724, 0.963181, 0.942244, 0.923017, 0.90608, 0.891381, 0.877957, 0.865492,
    0.854776, 0.844714, 0.835543, 0.827028, 0.819094, 0.811623, 0.804506, 0.7981,
    0.792694, 0.787222, 0.781869, 0.77645, 0.771312, 0.766128, 0.760936, 0.755641,
    0.750499, 0.745735, 0.74118, 0.736773, 0.732596, 0.728596, 0.724804, 0.720961,
    0.71751, 0.714109, 0.710818, 0.707797, 0.704781, 0.701984, 0.699186, 0.696473,
    0.693896, 0.694
]

final_loss = [
    2.950103, 2.685029, 2.399863, 2.19934, 2.039857, 1.919985, 1.819058, 1.735572,
    1.664016, 1.601277, 1.546897, 1.498681, 1.456372, 1.418558, 1.38585, 1.35577,
    1.328698, 1.303994, 1.281653, 1.260715, 1.241337, 1.223084, 1.206334, 1.190186,
    1.175023, 1.160653, 1.146659, 1.133185, 1.12065, 1.10842, 1.096772, 1.08529,
    1.074614, 1.064456, 1.05463, 1.045116, 1.035849, 1.027059, 1.018578, 1.01055,
    1.002758, 0.995402, 0.988263, 0.981328, 0.974917, 0.968623, 0.96267, 0.956919,
    0.951486, 0.95042
]

# X-axis: trial steps
steps = np.arange(2, 2 * len(start_loss) + 1, 2) #np.arange(1, len(start_loss) + 1)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, final_loss, label="Initial (Randomly Selected Parameters)", linewidth=4, color="#3A86FF")
plt.plot(steps, start_loss, label="Final (Agent Selected Parameters)", linewidth=4, color="black", linestyle='--')

# Aesthetics
plt.xlabel('Step', fontsize=24)
plt.ylabel('Loss', fontsize=24)
# plt.title('Loss Evolution Across Trials', fontsize=22)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

plt.savefig("/lustre/orion/stf218/proj-shared/brave/vllm_test/plots/loss.png")

label="Initial (Randomly Selected Parameters)"
label="Final (LLM-Guided Parameters)"
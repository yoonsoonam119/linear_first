import numpy as np
from networks.linear_network import LinearNetwork
from utils import get_random_regression_task, get_lambda_balanced, get_lambda_unbalanced,  get_ntk,kernel_distance,BlindColours, kernel_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
bc = BlindColours()
blind_colours = bc.get_colours()

in_dim = 20
hidden_dim = 20
out_dim = 2

batch_size = 10
training_steps = 4000
learning_rate = 0.01

lmda_values = np.linspace(-0.5, 0.5, 40)
scale_values = np.linspace(0.1, 50, 40)


ws_empirical = {}
w1w1s_empirical = {}
w2w2s_empirical = {}
losses_list_empirical = {}
ntks_empirical = {}
kernel_distance_ntk = {}
convergence_steps = {}
ntk_init = {}
ntk_final = {}
loss_empirical = {}
losses_empirical = {}
kernel_distances = {}

X, Y = get_random_regression_task(batch_size, in_dim, out_dim, Whiten=False)


for lmda in lmda_values:
    for scale in scale_values:
        init_w1, init_w2 = get_lambda_unbalanced(lmda, in_dim, hidden_dim, out_dim, scale=0.1)

        model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
        w1s, w2s, losses = model.train(X.copy(), Y.copy()/scale, training_steps, learning_rate)

        ws_empirical[(lmda, scale)] = [w2 @ w1 for (w2, w1) in zip(w2s, w1s)]
        ws_empirical[(lmda, scale)] = np.expand_dims(ws_empirical[(lmda, scale)], axis=1)
        losses_empirical[(lmda, scale)] = [np.linalg.norm(ws[0] @ X - Y, ord='fro')**2 for ws in ws_empirical[(lmda, scale)]]
        loss_empirical[(lmda, scale)] = np.linalg.norm(ws_empirical[(lmda, scale)][-1][0] @ X - Y, ord='fro')**2

        w1w1s_empirical[(lmda, scale)] = np.array([w1.T @ w1 for w1 in w1s])
        w2w2s_empirical[(lmda, scale)] = np.array([w2 @ w2.T for w2 in w2s])

        ntk_init[(lmda, scale)] = get_ntk(w1w1s_empirical[(lmda, scale)][0], w2w2s_empirical[(lmda, scale)][0], X, out_dim)
        ntk_final[(lmda, scale)] = get_ntk(w1w1s_empirical[(lmda, scale)][-1], w2w2s_empirical[(lmda, scale)][-1], X, out_dim)

        ntks = [get_ntk(w1w1s_empirical[(lmda, scale)][i], w2w2s_empirical[(lmda, scale)][i], X, out_dim) for i in range(training_steps)]
        kernel_distances[(lmda, scale)] = [kernel_distance(ntk_init[(lmda, scale)], ntks[i]) for i in range(training_steps)]

        kernel_distance_ntk[(lmda, scale)] = kernel_distance(ntk_init[(lmda, scale)], ntk_final[(lmda, scale)])


keys = list(kernel_distance_ntk.keys())
rows = sorted(set(key[0] for key in keys))  # Unique values of first parameter
cols = sorted(set(key[1] for key in keys))  # Unique values of second parameter


num_lmda_ticks = 3  # Adjust this for the desired number of ticks on the lambda axis
num_scale_ticks = 3  # Adjust this for the desired number of ticks on the scale axis

# Initialize an empty DataFrame with rows and columns
kernel_heatmap = pd.DataFrame(
    np.nan, index=rows, columns=cols
)

loss_heatmap = pd.DataFrame(
    np.nan, index=rows, columns=cols
)

lmda_tick_positions = np.linspace(0, kernel_heatmap.shape[0] - 1, num_lmda_ticks)
lmda_tick_labels = np.linspace(lmda_values.min(), lmda_values.max(), num_lmda_ticks)

scale_tick_positions = np.linspace(0, kernel_heatmap.shape[1] - 1, num_scale_ticks)
scale_tick_labels = np.linspace(scale_values.min(), scale_values.max(), num_scale_ticks)


lmda_ticks_labels = [-9, 0, 40]
scale_ticks_labels = [0, 10, 40]

# Define steps of interest
steps_of_interest = [1,4000]

# Initialize heatmaps for kernel distance and loss for each step
kernel_heatmaps = {step: pd.DataFrame(np.nan, index=rows, columns=cols) for step in steps_of_interest}
loss_heatmaps = {step: pd.DataFrame(np.nan, index=rows, columns=cols) for step in steps_of_interest}

# Populate the heatmaps
for step in steps_of_interest:
    for (param1, param2), distances in kernel_distances.items():
        if step <= len(distances):  # Ensure the step exists in the data
            kernel_heatmaps[step].loc[param1, param2] = distances[step - 1]  # Adjust for 0-based indexing

    for (param1, param2), losses in losses_empirical.items():
        if step <= len(losses):  # Ensure the step exists in the data
            loss_heatmaps[step].loc[param1, param2] = losses[step - 1]  # Adjust for 0-based indexing

for step in steps_of_interest:
    kernel_heatmaps[step] = np.log10(kernel_heatmaps[step] + 1e-6)  # Add a small constant to avoid log(0)
    loss_heatmaps[step] = np.log10(loss_heatmaps[step] + 1e-6)

# Use the scale of the last kernel heatmap and the first loss heatmap
kernel_scale = kernel_heatmaps[steps_of_interest[-1]]  # Last kernel heatmap
loss_scale = loss_heatmaps[steps_of_interest[0]]  # First loss heatmap

vmin_kernel = kernel_scale.min().min()
vmax_kernel = kernel_scale.max().max()
vmin_loss = loss_scale.min().min()
vmax_loss = loss_scale.max().max()

# Custom tick labels
x_tick_positions = [0, len(cols) // 2, len(cols) - 1]
x_tick_labels = [0, 10, 20]

y_tick_positions = [0, len(rows) // 2, len(rows) - 1]
y_tick_labels = [9, 0, -9]

# Create a figure with subplots for each step
fig, axes = plt.subplots(len(steps_of_interest), 2, figsize=(10, 10))  # Taller figure for better proportions

# Shared color maps for kernel distance and loss
cmap_kernel = 'coolwarm'
cmap_loss = 'coolwarm'

# Plot kernel distance and heatmaps
for i, step in enumerate(steps_of_interest):
    # Kernel distance heatmap
    sns.heatmap(kernel_heatmaps[step], 
                ax=axes[i, 0], 
                xticklabels=False,  # Disable default labels
                yticklabels=False,  # Disable default labels
                cmap=cmap_kernel, 
                cbar=False,  # Disable individual color bars
                vmin=vmin_kernel, vmax=vmax_kernel,  # Fix scale to the last kernel heatmap
                square=True)
    if i == 1:
        np.save('./data/scale_heatmap',kernel_heatmaps[step])
    axes[i, 0].set_title(f'Kernel Distance, t={step}', fontsize=10)
    axes[i, 0].set_xlabel('Absolute Scale', fontsize=8)
    axes[i, 0].set_ylabel('Relative Scale (λ)', fontsize=8)
    axes[i, 0].set_xticks(x_tick_positions)  # Custom x-tick positions
    axes[i, 0].set_xticklabels(x_tick_labels, fontsize=8)  # Custom x-tick labels
    axes[i, 0].set_yticks(y_tick_positions)  # Custom y-tick positions
    axes[i, 0].set_yticklabels(y_tick_labels, fontsize=8)  # Custom y-tick labels

cbar_ax_kernel = fig.add_axes([0.62, 0.3, 0.02, 0.4])  # Move closer to the kernel heatmaps
cbar_ax_loss = fig.add_axes([0.72, 0.3, 0.02, 0.4])  

# Create color bars
sm_kernel = plt.cm.ScalarMappable(cmap=cmap_kernel, norm=plt.Normalize(vmin=vmin_kernel, vmax=vmax_kernel))
sm_loss = plt.cm.ScalarMappable(cmap=cmap_loss, norm=plt.Normalize(vmin=vmin_loss, vmax=vmax_loss))

cbar_kernel = plt.colorbar(sm_kernel, cax=cbar_ax_kernel, orientation='vertical')
cbar_loss = plt.colorbar(sm_loss, cax=cbar_ax_loss, orientation='vertical')

# Add labels to color bars
cbar_kernel.set_label('Kernel Distance', fontsize=10)
cbar_loss.set_label('Loss', fontsize=10)

# Adjust layout to accommodate the wider spacing
plt.subplots_adjust(hspace=5)  # Increase vertical spacing between rows
plt.tight_layout(rect=[0, 0, 0.78, 1])  # Reduce the main plot area further to allow more space for color bars
plt.show()

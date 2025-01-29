import numpy as np
from utils import BlindColours
import matplotlib.pyplot as plt
import seaborn as sns
bc = BlindColours()
blind_colours = bc.get_colours()

scale_heatmap = np.load('./data/scale_heatmap.npy')

#scale
vmin_scale = scale_heatmap.min().min()
vmax_scale = scale_heatmap.max().max()
cols = len(scale_heatmap[0])
rows = len(scale_heatmap)


# Custom tick labels
x_tick_positions_scale = [0, cols // 2, cols - 1]
x_tick_labels_scale = [0, 25, 50]

y_tick_positions_scale = [0, rows // 2, rows - 1]
y_tick_labels_scale = [5, 0, -5]

# Create a figure with subplots for each step
fig, ax = plt.subplots(1,1, figsize=(7, 7))  # Taller figure for better proportions

# Shared color maps for kernel distance and loss
cmap_kernel = 'coolwarm'
cmap_loss = 'coolwarm'

# Kernel distance heatmap

sns.heatmap(scale_heatmap,
            ax=ax,
            xticklabels=False,  # Disable default labels
            yticklabels=False,  # Disable default labels
            cmap=cmap_kernel,
            cbar=False,  # Disable individual color bars
            vmin=vmin_scale, vmax=vmax_scale,  # Fix scale to the last kernel heatmap
            square=True)
ax.set_xlabel('Weight-Target-Ratio', fontsize=18)
ax.set_ylabel('Layer Imbalance ' +  r'$(\sigma_1 - \sigma_2)$', fontsize=18)
ax.set_xticks(x_tick_positions_scale)  # Custom x-tick positions
ax.set_xticklabels(x_tick_labels_scale, fontsize=18)  # Custom x-tick labels
ax.set_yticks(y_tick_positions_scale)  # Custom y-tick positions
ax.set_yticklabels(y_tick_labels_scale, fontsize=18)  # Custom y-tick labels

# Add shared vertical color bars with increased spacing
cbar_ax_kernel = fig.add_axes([0.1, 0.98, 0.6, 0.02])  # Move closer to the kernel heatmaps

# Create color bars
sm_scale = plt.cm.ScalarMappable(cmap=cmap_kernel, norm=plt.Normalize(vmin=vmin_scale, vmax=vmax_scale))

cbar_scale = plt.colorbar(sm_scale, cax=cbar_ax_kernel, orientation='horizontal')

# Add labels to color bars
cbar_scale.set_label('Kernel Distance (Log Scale)', fontsize=20, labelpad=10)

# Adjust layout to accommodate the wider spacing
plt.subplots_adjust(hspace=5)  # Increase vertical spacing between rows
plt.tight_layout(rect=[0, 0, 0.78, 1])  # Reduce the main plot area further to allow more space for color bars
plt.savefig('plot/scale_figure.pdf', format='pdf', bbox_inches='tight' ,dpi=300)
plt.show()

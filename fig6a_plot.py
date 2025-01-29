import numpy as np
from utils import BlindColours
import matplotlib.pyplot as plt
import seaborn as sns
bc = BlindColours()
blind_colours = bc.get_colours()


imba_heatmap = np.load('./data/imba_heatmap.npy')

#imba
vmin_imba = imba_heatmap.min().min()
vmax_imba = imba_heatmap.max().max()
cols = len(imba_heatmap[0])
rows = len(imba_heatmap)

# Custom tick labels
x_tick_positions_imba = [0, cols // 2, cols - 1]
x_tick_labels_imba = [0, 10, 20]

y_tick_positions_imba = [0, rows // 2, rows - 1]
y_tick_labels_imba = [9, 0, -9]

# Create a figure with subplots for each step
fig, ax = plt.subplots(1,1, figsize=(7, 7))  # Taller figure for better proportions

# Shared color maps for kernel distance and loss
cmap_kernel = 'coolwarm'
cmap_loss = 'coolwarm'

# Kernel distance heatmap

sns.heatmap(imba_heatmap,
            ax=ax,
            xticklabels=False,  # Disable default labels
            yticklabels=False,  # Disable default labels
            cmap=cmap_kernel,
            cbar=False,  # Disable individual color bars
            vmin=vmin_imba, vmax=vmax_imba,  # Fix scale to the last kernel heatmap
            square=True)
ax.set_xlabel('Weight-Target-Ratio', fontsize=18)
ax.set_ylabel('Layer Imbalance ' +  r'$\lambda$', fontsize=18)
ax.set_xticks(x_tick_positions_imba)  # Custom x-tick positions
ax.set_xticklabels(x_tick_labels_imba, fontsize=18)  # Custom x-tick labels
ax.set_yticks(y_tick_positions_imba)  # Custom y-tick positions
ax.set_yticklabels(y_tick_labels_imba, fontsize=18)  # Custom y-tick labels

# Add shared vertical color bars with increased spacing
cbar_ax_kernel = fig.add_axes([0.1, 0.98, 0.6, 0.02])  # Move closer to the kernel heatmaps


# Create color bars
sm_imba = plt.cm.ScalarMappable(cmap=cmap_kernel, norm=plt.Normalize(vmin=vmin_imba, vmax=vmax_imba))

cbar_imba = plt.colorbar(sm_imba, cax=cbar_ax_kernel, orientation='horizontal')

# Add labels to color bars
cbar_imba.set_label('Kernel Distance (Log Scale)', fontsize=20, labelpad=10)

# Adjust layout to accommodate the wider spacing
plt.subplots_adjust(hspace=5)  # Increase vertical spacing between rows
plt.tight_layout(rect=[0, 0, 0.78, 1])  # Reduce the main plot area further to allow more space for color bars
plt.savefig('plot/imba_figure.pdf', format='pdf', bbox_inches='tight' ,dpi=300)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
import argparse

CYAN = '\033[96m'
RESET = '\033[0m'

parser = argparse.ArgumentParser()
parser.add_argument(
    "csv_file",
    type=str,
    help="Path to the input CSV file"
)
parser.add_argument(
    "--top",
    type=int,
    default=0,
    help="Number of top Pareto points to highlight for each axis")
args = parser.parse_args()

# Load CSV data
original_data = pd.read_csv(args.csv_file)

#====================[ parameterization ]========================
# step 1. Define the labels for the axes: works for 3 or 4 variables
labels = ['Accuracy', 'FLOP', 'aPE', 'ECE']
# step 2. Define the direction for each label
#    Note that "1" means higher is better, "-1" means lower is better
direction = np.array([1, -1, 1, -1])

# Filter the dataset
# step 3. Adjust the variable thresholds as needed to filter out unwanted data
score_threshold = -1e+18  # Filter out extremely low scores
data = original_data[original_data['score'] > score_threshold]

flop_threshold = 2E+9
data = data[original_data['FLOP'] < flop_threshold]
#ece_threshold = 0.09
#ece_down_threshold = 0.05
#data = data[original_data['ECE'] < ece_threshold]
#data = data[original_data['ECE'] > ece_down_threshold]
#==========================================================================

# Extract the relevant columns
X = data[labels].values

# Adjust the points based on direction
X_adjusted = X * direction  # Flip the sign of minimization criteria

# Function to find Pareto front
def pareto_frontier(points):
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if all(q >= p) and any(q > p):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return np.array(pareto)

# Get Pareto-optimal points
pareto_points_adjusted = pareto_frontier(X_adjusted)

# Convert Pareto points back to original scale
pareto_points = pareto_points_adjusted * direction

# Get the top N Pareto points for each axis
# Convert to DataFrame for filtering
pareto_df = pd.DataFrame(pareto_points, columns=labels)

max_points = []

for label, d in zip(labels, direction):
    if d == 1:
        max_points.append(pareto_df.nlargest(1, label))
    else:
        max_points.append(pareto_df.nsmallest(1, label))

optimal_per_axis = pd.concat(max_points).drop_duplicates()

if args.top > 0:
    # Filter N points for each axis and merge
    top_N = []
    for label, d in zip(labels, direction):
        if d == 1:
            top_N.append(pareto_df.nlargest(args.top, label))
        else:
            top_N.append(pareto_df.nsmallest(args.top, label))

    pareto_filtered = pd.concat(top_N).drop_duplicates()

    # Convert back to NumPy array
    pareto_points = pareto_filtered.values

######################################


# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Scatter all points with a lower alpha to make Pareto points stand out
# Scatter all points with a lower alpha
if len(labels) > 3:
    sc = ax.scatter(data[labels[0]],
                    data[labels[1]],
                    data[labels[2]],
                    s=80,
                    alpha=0.6,
                    c=data[labels[3]], cmap='coolwarm',
                    label='Configurations')
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label(labels[3])
else:
    ax.scatter(data[labels[0]],
               data[labels[1]],
               data[labels[2]],
               s=80,
               color='blue',
               alpha=0.2,
               label='Configurations')

if args.top > 0:
    top_label = f' (top {args.top})'
else:
    top_label = ''


ax.scatter(pareto_points[:, 0],
            pareto_points[:, 1],
            pareto_points[:, 2],
            s=200,
            facecolors='none',
            edgecolors='black',
            linewidth=1.5,
            marker='o',
            alpha=0.9,
            label='Pareto-optimal')

# Scatter optimal points per axis
ax.scatter(optimal_per_axis[labels[0]],
           optimal_per_axis[labels[1]],
           optimal_per_axis[labels[2]],
           alpha=0.8,
           facecolors='none',
           edgecolors='green',
           linewidth=2,
           marker='X',
           s=300,
           label='Optimal solutions (single axis)')


if len(pareto_points) >= 4:  # Delaunay requires at least 4 points
    tri = Delaunay(pareto_points[:, :2])  # Use first two columns for triangulation

    # Plot the smoother surface
    ax.plot_trisurf(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
                    triangles=tri.simplices, cmap='inferno', alpha=0.1, edgecolor='black')

# Labels
ax.grid(True)
ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_zlabel(labels[2])
ax.legend()

# print the pareto points
print(f"{CYAN}--[optimal points]---{RESET}")
print(optimal_per_axis)

# print the optimal points
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(f"{CYAN}--[pareto points]---{RESET}")
print(pd.DataFrame(pareto_points, columns=labels))



plt.show()

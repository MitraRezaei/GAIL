import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import numpy as np

# Load the data from the csv file
df = pd.read_csv('./data/vadere_csv.csv').replace([np.inf, -np.inf], 100).astype(float)
states = df[['x_world', 'y_world']].to_numpy()
actions = df[['w', 'v']].to_numpy()
frames = df[['frame']].astype(int)
frames_group = frames.groupby('frame')['frame'].unique()

fig, ax = plt.subplots(figsize=(8, 6))

# Define the obstacle coordinates
obstacle_coordinates = [
    [(0.5, 13.0), (0.5, 19.5), (7.0, 19.5), (7.0, 13.0)],       # obstacle 1
    [(0.5, 0.5), (0.5, 7.0), (7.0, 7.0), (7.0, 0.5)],           # obstacle 2
    [(13.0, 0.5), (13.0, 7.0), (19.5, 7.0), (19.5, 0.5)],       # obstacle 3
    [(13.0, 13.0), (13.0, 19.5), (19.5, 19.5), (19.5, 13.0)],   # obstacle 4
    [(9.0, 11.0), (9.0, 12.0), (10.0, 12.0), (10.0, 11.0)],     # obstacle 5
    [(11.0, 9.0), (11.0, 10.0), (12.0, 10.0), (12.0, 9.0)]      # obstacle 8
]

obstacle_patches = []
for coords in obstacle_coordinates:
    obstacle_patches.append(Polygon(coords, closed=True, alpha=0.5, facecolor='black'))


def update(i):
    ax.clear()
    # Add the obstacle patches to the axis
    for patch in obstacle_patches:
        ax.add_patch(patch)

    index = np.where(frames == i)
    frame_pedestrians = []
    for j in index[0]:
        frame_pedestrians.append(df['id'][j])
    main_pedestrian = min(frame_pedestrians)

    # for idx, pedestrian in df.iterrows():
    for idx in index[0]:
        if df['id'][idx] == main_pedestrian:
            ax.scatter(df['x_world'][idx], df['y_world'][idx], c='blue', marker='o')
            ax.annotate(int(df['id'][idx]), (df['x_world'][idx], df['y_world'][idx]))
        else:
            ax.scatter(df['x_world'][idx], df['y_world'][idx], c='red', marker='o')
            ax.annotate(int(df['id'][idx]), (df['x_world'][idx], df['y_world'][idx]))
    ax.set_xlim([min(df['x_world'])-5, max(df['x_world'])+5])
    ax.set_ylim([min(df['y_world'])-5, max(df['y_world'])+5])
    ax.set_xlabel('x_world')
    ax.set_ylabel('y_world')
    ax.set_title(f'Frame {i}')


frame_list = []
for i, val in enumerate(frames_group):
    frame_list.append(frames_group[val[0]][0])
    # update(frames_group[val[0]][0])

anim = FuncAnimation(fig, update, frames=frame_list, interval=50, repeat=True)
anim.save('Simulation.avi', fps=30, codec='mpeg4')
plt.show()


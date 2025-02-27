import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_ray_paths(all_path_segments, all_hits, xlim=(-30,30), ylim=(-40,5), zlim=(-30,30)):
    """
    Visualize ray paths in 3D with segments colored by type and intensity

    Parameters:
    -----------
    all_path_segments : list
        List of Segment objects containing ray path information
    all_hits : list
        List of Hit objects containing intersection points
    xlim, ylim, zlim : tuple
        Plot axis limits (min, max) for each dimension
    """
    # Create 3D figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Define colormaps for different ray types
    primary_color = plt.cm.Greens
    reflected_color = plt.cm.Reds
    refracted_color = plt.cm.Blues

    # Plot segments with colors based on type and depth
    num_points = 50
    max_seen_depth = max(segment.depth for segment in all_path_segments)

    for segment in all_path_segments:
        points = segment.get_points()
        t = np.linspace(0, 1, num_points)

        # Generate interpolated points along segment
        segment_points = np.outer(1-t, points[0]) + np.outer(t, points[1])
        distances = np.linalg.norm(segment_points - points[0], axis=1)
        intensities = segment.get_intensity_at_distance(distances)

        # Choose colormap based on ray type
        if segment.ray_type == 'primary':
            colormap = primary_color
        elif segment.ray_type == 'reflected':
            colormap = reflected_color
        else:  # refracted
            colormap = refracted_color

        # Calculate color based on depth
        depth_color = (segment.depth + 1) / (max_seen_depth + 1)

        # Plot line segments with intensity-based alpha
        for i in range(len(segment_points)-1):
            color = colormap(depth_color)
            ax.plot3D(segment_points[i:i+2,0],
                     segment_points[i:i+2,1],
                     segment_points[i:i+2,2],
                     color=color,
                     alpha=intensities[i] * 0.5,
                     linewidth=1.5 * (1 - depth_color + 0.5))

    # Plot intersection points with depth-based size
    max_size = 100
    min_size = 20

    hit_points = np.array([hit[1] for hit in all_hits if hit is not None])


    ax.scatter(hit_points[:, 0], hit_points[:,1], hit_points[:,2])
    # Plot settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title("3D Ray Tracing Visualization")
    ax.grid(True)

    # Add colorbar for intensity
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='Greys')
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Intensity')

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

    return fig, ax

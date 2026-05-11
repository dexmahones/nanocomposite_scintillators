import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.animation as animation

output_dir = "./PET_anim_results"
# Physical constants
HALF_LIFE_MIN = 109.8
DECAY_CONSTANT = np.log(2) / HALF_LIFE_MIN

# Define Detector Geometry
num_pixels = 16
radius = 750         # Distance from center to pixels
pixel_size = (150, 300) # Width, Height of each scintillator
center_x, center_y = 0,0

# Calculate the angular width (half-width) each pixel spans in radians
half_span = (pixel_size[0] / radius) / 2

# Store the center angle for each of the 8 pixels
pixel_angles = [(2 * np.pi / num_pixels) * i for i in range(num_pixels)]

def get_hit_pixels(x_e, y_e, gamma_theta, pixels, p_width, radius):
    """
    Checks if a gamma ray starting at (x_e, y_e) with angle gamma_theta
    intersects the front face of any pixel.
    """
    hit_indices = []
    
    # Check both back-to-back directions
    thetas_to_check = [gamma_theta % (2 * np.pi), (gamma_theta + np.pi) % (2 * np.pi)]
    
    for i, p_angle in enumerate(pixel_angles):
        # Coordinates of the two inner corners of the pixel
        dx_edge = (p_width / 2) * np.sin(p_angle)
        dy_edge = (p_width / 2) * np.cos(p_angle)
        
        c1 = (center_x + radius * np.cos(p_angle) - dx_edge, 
              center_y + radius * np.sin(p_angle) + dy_edge)
        c2 = (center_x + radius * np.cos(p_angle) + dx_edge, 
              center_y + radius * np.sin(p_angle) - dy_edge)
        
        # Calculate angles of corners relative to the emission point (x_e, y_e)
        angle1 = np.arctan2(c1[1] - y_e, c1[0] - x_e) % (2 * np.pi)
        angle2 = np.arctan2(c2[1] - y_e, c2[0] - x_e) % (2 * np.pi)
        
        # Check if gamma_theta is between angle1 and angle2
        # We need to handle the 0/2pi wrap-around carefully
        for gt in thetas_to_check:
            # Find the angular span
            a_min, a_max = min(angle1, angle2), max(angle1, angle2)
            
            if a_max - a_min > np.pi: # The span crosses the 0/2pi line
                if gt >= a_max or gt <= a_min:
                    hit_indices.append(i)
            else:
                if a_min <= gt <= a_max:
                    hit_indices.append(i)
                    
    return list(set(hit_indices))

def plot_pixels(pixels, ax, show_hits = True):
    # Plot each pixel as a rectangle
    rectangles = []
    for i, p in enumerate(pixels):
        # Determine color based on hit status
        color = 'cyan' if (p['hit'] and show_hits) else 'blue'
        alpha = 0.4 if (p['hit'] and show_hits) else 0.1
        
        # Calculate position angle
        angle_rad = (2 * np.pi / num_pixels) * i
        angle_deg = np.degrees(angle_rad)

        # Center position of the pixel
        px = center_x + radius * np.cos(angle_rad)
        py = center_y + radius * np.sin(angle_rad)

        # Create the rectangle patch
        # Note: (x_min, y_min) is the bottom-left corner
        rect = patches.Rectangle(
            (px - pixel_size[0]/2, py - pixel_size[1]/2), # Center the box on (px, py)
            pixel_size[0], 
            pixel_size[1],
            angle = angle_deg - 90, 
            rotation_point='center',
            linewidth=2,
            edgecolor='none',
            facecolor=color,
            alpha=alpha,
            label=f'Pixel {i}'
        )
        rectangles.append(rect)
        ax.add_patch(rect)
    return rectangles

img = Image.open('icon.png')

mask_img = 255 - np.array(img.convert('L'))

source_activity = mask_img / np.sum(mask_img.flatten())
flat_pdf = source_activity.ravel()
indices = np.arange(flat_pdf.size)
ny,nx = source_activity.shape
xs = np.arange(-nx//2,nx//2)
ys = np.arange(-ny//2,ny//2)

# Simulation parameters
initial_activity = 5000  # Initial particles to simulate at t=0
mins_per_step = 0.1       # Time interval for each step in mins
total_steps = 1000
gamma_length = 2000

# Simulate emissions over time
current_particles = initial_activity
total_emissions = []
emission_events = np.zeros_like(source_activity)
frames = [] # This will hold the "Artists" for each frame
coincidence_detections = []
coincident_xs = []
coincident_ys = []
save_frames = True

if not save_frames:
    fig, ax = plt.subplots(1,1,figsize = (7,7))
    ax.set_xlim(-1000,1000)
    ax.set_ylim(-1000,1000)
    ax.axis('off')
    ax.set_aspect('equal')

for t_step in range(total_steps):
    if save_frames:
        fig, ax = plt.subplots(1,1,figsize = (7,7))
        ax.set_xlim(-1000,1000)
        ax.set_ylim(-1000,1000)
        ax.axis('off')
        ax.set_aspect('equal')

    frame_artists = []
    t = t_step * mins_per_step
    gamma_rays = []
    pixels = []
    for i in range(num_pixels):
        angle = (2 * np.pi / num_pixels) * i
        # Position the pixel center on the ring
        px = center_x + radius * np.cos(angle)
        py = center_y + radius * np.sin(angle)
        # Define bounds (Axis-Aligned for simplicity)
        p = {
            'hit': False
        }
        pixels.append(p)
    # Calculate expected number of decays in this interval
    # N_decayed = N_initial * (1 - e^(-lambda * delta_t))
    expected_decays = current_particles * (1 - np.exp(-DECAY_CONSTANT * mins_per_step))
    
    # Only a fraction of these decays emit a positron
    positron_emissions = int(expected_decays) # Set this to one to simulate events at a time
    positron_emissions = 1
    
    # Sample coordinates using the PDF from the previous step
    sampled_indices = np.random.choice(indices, size=positron_emissions, p=flat_pdf)
    
    # Convert and store points
    rows, cols = np.unravel_index(sampled_indices, source_activity.shape)
    xps,yps = xs[cols],ys[rows]

    # Add back-to-back gammas
    for x, y in zip(xps, yps):
        # Pick a random emission angle
        theta = np.random.uniform(0, 2 * np.pi)
        # Calculate the unit vector components
        dx = np.cos(theta) * gamma_length
        dy = np.sin(theta) * gamma_length
        
        # Create the two back-to-back endpoints
        # Line goes from (x - dx, y - dy) to (x + dx, y + dy)
        # This represents both gammas passing through the emission point
        start_point = (x - dx, y - dy)
        end_point = (x + dx, y + dy)
        
        gamma_rays.append((start_point, end_point))

        hit_indices = get_hit_pixels(x,y,theta,pixels,pixel_size[0],radius)
        for idx in hit_indices:
            pixels[idx]['hit'] = True
            if len(hit_indices)==2:
                coincidence_detections.append((start_point, end_point))
                coincident_xs.append((start_point[0]+end_point[0])/2)
                coincident_ys.append((start_point[1]+end_point[1])/2)

    emission_events[rows,cols] += 1
    # ax.pcolormesh(xs,ys,emission_events,cmap = "binary")
    pts = ax.scatter(xps,yps, color = 'r', alpha = 0.3)
    frame_artists.append(pts)

    for start, end in gamma_rays:
        gc = "cyan" if len(hit_indices)==2 else "red"
        lw = 2 if len(hit_indices)==2 else 1
        line, = ax.plot([start[0], end[0]], [start[1], end[1]], color=gc, alpha=0.3, lw=lw)
        frame_artists.append(line)

    for start, end in coincidence_detections:
        cline, = ax.plot([start[0], end[0]], [start[1], end[1]], color='k', alpha=0.05, lw=2)
        frame_artists.append(cline)

    cpts = ax.scatter(coincident_xs,coincident_ys,marker = '.', color = 'k', alpha = 0.05)
    frame_artists.append(cpts)
    # Plot each pixel as a rectangle
    rectangles = []
    for i, p in enumerate(pixels):
        # Determine color based on hit status
        color = 'cyan' if p['hit'] else 'blue'
        alpha = 0.4 if p['hit'] else 0.1
        
        # Calculate position angle
        angle_rad = (2 * np.pi / num_pixels) * i
        angle_deg = np.degrees(angle_rad)

        # Center position of the pixel
        px = center_x + radius * np.cos(angle_rad)
        py = center_y + radius * np.sin(angle_rad)

        # Create the rectangle patch
        # Note: (x_min, y_min) is the bottom-left corner
        rect = patches.Rectangle(
            (px - pixel_size[0]/2, py - pixel_size[1]/2), # Center the box on (px, py)
            pixel_size[0], 
            pixel_size[1],
            angle = angle_deg - 90, 
            rotation_point='center',
            linewidth=2,
            edgecolor='none',
            facecolor=color,
            alpha=alpha,
            label=f'Pixel {i}'
        )
        frame_artists.append(ax.add_patch(rect))
    
    frames.append(frame_artists)
    if save_frames:
        plt.savefig(f'{output_dir}/frame_{t_step}.svg', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    total_emissions.append((start_point,end_point))
    
    # Update current particle count for next step
    current_particles -= expected_decays

if not save_frames:
    # Create the animation object
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # Save directly as a GIF using the Pillow writer
    filename = "pet_anim.gif"
    ani.save(filename, writer='pillow', fps=10)
    plt.close()
    print(f"Animation saved as {filename}")

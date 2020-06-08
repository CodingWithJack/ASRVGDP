# Imports needed libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio

""" Instructions for use:
    
Change any of the variables directly below this.

If you use the default media paths, make sure you have have a directory called
Media, and a directory within that called DroneGIF.

Simply run the code to use, no functions are required. """

# User variables
aw = 500  # Avalance width
al = 500  # Avalance length
mld = 50  # Maximum launch distance away from avalance field
sim_time = 5000  # How long the simulation goes on for (prevents loop errors)
sim_step = 0.1  # Seconds between each data point, lower gives a higher resolution

# Simulation variables
no_drones = 1  # Number of drones
start_times = []  # If empty picks start times 1 min apart on average (array on integers)
start_positions = []  # If empty picks a random start positions (array of np arrays)
drones_start_in_avalanche = False  # Can drones start in the avalanche area
no_victims = 0  # The number of victims in the simulation
victim_pos = []  # If empty chooses random victim locations (array of np arrays)

# Drone variables
speed = 13  # Drone speed in m/s
beacon_range = 70  # Beacon range in m
# Recommended 1.6 and 0.64
wavelength = 1.84 * beacon_range  # Peak to peak distance of sine sweep in m
edge_dist = 0.52 * beacon_range  # Minimum distance from the edge of the sweep area
avoid_sens = 12  # Distance at which drones will begin making evasive manouevres

# Output preferences
show_swept_area = True # Shows the area covered in light blue
show_grid = True
show_time = True
result_name = 'DroneSweep'  # Name of saved gif files and pngs.
media_path = ''  # Path for output images
plot_gif = False  # Plot gif or simply static image?
gif_frame_speed = 1  # Sim time between frames in seconds (gif has 10 fps)
gif_end_freeze_time = 1  # Length of static image at end of gif
gif_frame_storage = '' # Path for output gif frames

# Other options
y_offset = 0 # Should be left at 0
arrow_plot = False

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #
" SIMULATION CODE BEGINS HERE - DO NOT CHANGE VARIABLES BELOW THIS LINE!  "
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #

# Code run by each drone
class drone():

    numDrones = 0
    drones = [] # Postion of all the drones
    sweep_update = [] # set to 1 if a drone needs to sweep update
    top_branch = [] # Drone IDs of drones sweeping down
    bottom_branch = [] # Drone IDs of drones sweeping up
    avoid_areas = [] # Areas to ignore signal due to victim already found

    def __init__(self, pos, aw, al):
        # Private
        self.droneId = drone.numDrones
        self.state = 'initialize'  # initialize, sweep, search
        self.branch = 'nil'
        self.vic_pos = np.array((0, 0))
        # Public
        drone.numDrones += 1
        self.aw = aw
        self.al = al
        self.pos = pos
        self.drones.append(pos)
        self.sweep_update.append(0)
        self.sweep_dir = (0, 0)
    
    def collision_avoid(self, heading):
        for i in range(len(self.drones)):
            if i != self.droneId:
                vec_from_drone = self.pos - self.drones[i]
                dist = np.linalg.norm(vec_from_drone)
                if dist < avoid_sens:
                    norm_vec = vec_from_drone / dist
                    heading = heading + norm_vec * (avoid_sens - dist) / (avoid_sens - 1)
        if np.linalg.norm(heading) > 0.1:
            return heading / np.linalg.norm(heading)
        else:
            return np.array((0, 0))

    def heading(self):
        if self.state == 'initialize':
            vec_to_target = (0, 0)
            # Find vector to nearest effective corner
            corners = []
            branch = []
            corner_x = self.pos[0]
            # Nearest effective corner to the top
            if len(self.top_branch) > 0:
                if np.abs(corner_x) > self.aw / 2 - edge_dist:
                    corner_x = (self.aw / 2 - edge_dist) * np.abs(corner_x) / corner_x
                corner_y = self.drones[self.top_branch[0]][1]
                corners.append(np.array((corner_x, corner_y)))
                branch.append('top')
            else:
                corners.append(np.array((self.aw / 2 - edge_dist, self.al / 2)))
                branch.append('top')
                corners.append(np.array((-self.aw / 2 + edge_dist, self.al / 2)))
                branch.append('top')
            # Nearest effective corner to the bottom
            if len(self.bottom_branch) > 0:
                if np.abs(corner_x) > self.aw / 2 - edge_dist:
                    corner_x = (self.aw / 2 - edge_dist) * np.abs(corner_x) / corner_x
                corner_y = self.drones[self.bottom_branch[0]][1]
                corners.append(np.array((corner_x, corner_y)))
                branch.append('bottom')
            else:
                corners.append(np.array((self.aw / 2 - edge_dist, -self.al / 2)))
                branch.append('bottom')
                corners.append(np.array((-self.aw / 2 + edge_dist, -self.al / 2)))
                branch.append('bottom')
            # Find nearest corner
            corner_distances = []
            for i in range(len(corners)):
                corner_distances.append(np.linalg.norm(corners[i] - self.pos))
            nearest_corner = corners[corner_distances.index(min(corner_distances))]
            self.branch = branch[corner_distances.index(min(corner_distances))]
            vec_to_target = nearest_corner - self.pos
            # Change state to sweep
            if np.linalg.norm(vec_to_target) < 2 or y_offset != 0:
                self.state = 'sweep'
                if self.branch == 'bottom':
                    dir_y = 1
                    self.bottom_branch.append(self.droneId)
                    if len(self.bottom_branch) > 1:
                        x_positions = []
                        Z = []
                        for i in range(len(self.bottom_branch)):
                            x_positions.append(self.drones[self.bottom_branch[i]][0])
                            # Index one array based on another
                            Z = [x for _,x in sorted(zip(x_positions, self.bottom_branch))]
                            # Let other drones know
                            if self.bottom_branch[i] != self.droneId:
                                self.sweep_update[self.bottom_branch[i]] = 1
                        drone.bottom_branch = Z
                else:
                    dir_y = -1
                    self.top_branch.append(self.droneId)
                    if len(self.top_branch) > 1:
                        x_positions = []
                        Z = []
                        for i in range(len(self.top_branch)):
                            x_positions.append(self.drones[self.top_branch[i]][0])
                            Z = [x for _,x in sorted(zip(x_positions, self.top_branch))]
                            if self.top_branch[i] != self.droneId:
                                  self.sweep_update[self.top_branch[i]] = 1
                        drone.top_branch = Z
                # Set sweep direction
                self.sweep_dir = np.array((self.sweep_ideal_dir(dir_y), dir_y))
            # Return heading
            return self.collision_avoid(vec_to_target / np.linalg.norm(vec_to_target))
        elif self.state == 'sweep':
            # If new sweep pattern started want to travel smallest distance
#            if self.sweep_update[self.droneId] == 1:
#                self.sweep_dir = np.array((
#                    self.sweep_ideal_dir(self.sweep_dir[1]),
#                    self.sweep_dir[1]))
#                self.sweep_update[self.droneId] = 0
            local_drones = 1
            if self.branch == 'bottom':
                local_drones = len(self.bottom_branch)
            else:
                local_drones = len(self.top_branch)
            E_aw = self.aw / local_drones
            if E_aw > beacon_range:
                if self.branch == 'bottom':
                    E_x = E_aw * self.bottom_branch.index(self.droneId)
                else:
                    E_x = E_aw * self.top_branch.index(self.droneId)
                # The ideal x and y coordinates
                y_dist = np.abs(-(self.al / 2) * self.sweep_dir[1] - self.pos[1]) + y_offset
                x_dist = self.sine_disp(y_dist, E_aw) * self.sweep_dir[0]
                x_ideal = x_dist - self.aw / 2 + E_x + E_aw / 2
                # The difference between that and where I am
                drone_to_ideal = np.array((x_ideal - self.pos[0], 0))
                # The wanted velocity
                y_vel = self.sweep_dir[1]
                x_vel = self.sine_velocity(y_dist, E_aw)
                total_velocity = np.array((x_vel, y_vel))
                # heading direction
                heading = drone_to_ideal + total_velocity
                # return unit vector
                return self.collision_avoid(heading / np.linalg.norm(heading))
        elif self.state == 'course_search':
            heading = self.vic_pos - self.pos
            return self.collision_avoid(heading / np.linalg.norm(heading))
        elif self.state == 'done':
            return self.collision_avoid(np.array((0, 0)))
    
    def sweep_ideal_dir(self, dir_y):
        local_drones = 1
        if self.branch == 'bottom':
            local_drones = len(self.bottom_branch)
        else:
            local_drones = len(self.top_branch)
        E_aw = self.aw / local_drones
        if self.branch == 'bottom':
            E_x = E_aw * self.bottom_branch.index(self.droneId)
        else:
            E_x = E_aw * self.top_branch.index(self.droneId)
        # The ideal x and y coordinates
        y_dist = np.abs(-(self.al / 2) * dir_y - self.pos[1])
        distances = []
        dir_x = [-1, 1]
        for i in dir_x:
            x_dist = self.sine_disp(y_dist, E_aw) * i
            x_ideal = x_dist - self.aw / 2 + E_x + E_aw / 2
            # The difference between that and where I am
            distances.append(np.abs(x_ideal - self.pos[0]))
        return dir_x[distances.index(min(distances))]

    def sine_disp(self, x, W):
        return ((W / 2) - edge_dist) * (-np.cos(2 * np.pi * x / wavelength))

    def sine_velocity(self, x, W):
        return (np.pi / wavelength) * (W - 2 * edge_dist) * np.sin(2 * np.pi * x / wavelength)
    
    def past_other_drones(self):
        past_others = False
        if self.sweep_dir[1] == 1:
            worst_case = -self.al
            for i in self.top_branch:
                if self.drones[i][1] > worst_case:
                    worst_case = self.drones[i][1]
            if self.pos[1] > worst_case:
                past_others = True
        else:
            worst_case = self.al
            for i in self.bottom_branch:
                if self.drones[i][1] < worst_case:
                    worst_case = self.drones[i][1]
            if self.pos[1] < worst_case:
                past_others = True
        return past_others

    def update(self, timestep):
        # Moves the drone
        travel = speed * timestep
        self.pos = self.pos + self.heading() * travel
        # Stop if on top of a victim
        if self.state == 'course_search':
            if np.linalg.norm(self.pos - self.vic_pos) < 2:
                global victim_found
                victim_found = True
                self.state = 'done'
        # Does the beacon detect any victims?
        elif no_victims > 0 and not self.state == 'done':
            # Make sure not going to a victim already found
            avoid_detection = False
            for i in self.avoid_areas:
                if np.linalg.norm(self.pos - i) < beacon_range + 2:
                    avoid_detection = True
            # Change state to find new victim & alert other drones
            if not avoid_detection:
                for i in victim_pos:
                    if np.linalg.norm(self.pos - i) < beacon_range:
                        if self.state == 'sweep':
                            if self.branch == 'bottom':
                                del self.bottom_branch[self.bottom_branch.index(self.droneId)]
                                for o in self.bottom_branch:
                                    self.sweep_update[o] = 1
                            else:
                                del self.top_branch[self.top_branch.index(self.droneId)]
                                for o in self.top_branch:
                                    self.sweep_update[o] = 1
                        drone.avoid_areas.append(i)
                        self.vic_pos = i
                        self.state = 'course_search'
        # Stop if full avalanche area swept
        if self.state == 'sweep':
            if len(self.top_branch) > 0 and len(self.bottom_branch) > 0:
                if self.past_other_drones():
                    self.state = 'done'
            if np.abs(self.pos[1]) > np.abs(self.al / 2) + 2:
                self.state = 'done'
        self.drones[self.droneId] = self.pos

    def reset():
        drone.numDrones = 0
        drone.drones = []
        drone.top_branch = []
        drone.bottom_branch = []

# Resets variables
drone.reset()
drone_objs = []
end_sim = 0
histories = {}
steps = int(np.ceil(sim_time / sim_step))
images = []
swept_area = []
max_length = max([aw/2 + mld, al/2 + mld])
victim_found = False

# Generate the plot
def setup_plt():
    # Create final plot
    plt.figure(figsize=(10, 10))
    plt.plot([-aw/2, aw/2, aw/2, -aw/2, -aw/2], [-al/2, -al/2, al/2, al/2, -al/2])
    max_length = max([aw/2 + mld, al/2 + mld])
    plt.axis([-max_length, max_length, -max_length, max_length])
    if show_grid:
        plt.grid()

if arrow_plot:
    density = 20
    diff = aw / (density-1)
    X, Y = np.meshgrid(np.arange(-aw/2, aw/2 + diff, aw/(density - 1)),
                       np.arange(-al/2, al/2 + diff, al/(density - 1)))
    def get_UV(pos):
        def sine_disp(x, W):
            return ((W / 2) - edge_dist) * (-np.cos(2 * np.pi * x / wavelength))
        def sine_velocity(x, W):
            return (np.pi / wavelength) * (W - 2 * edge_dist) * np.sin(2 * np.pi * x / wavelength)
        # The ideal x and y coordinates
        y_dist = np.abs(-(al / 2) - pos[1])
        x_ideal = sine_disp(y_dist, aw)
        # The difference between that and where I am
        drone_to_ideal = np.array((x_ideal - pos[0], 0))
        # The wanted velocity
        y_vel = 1
        x_vel = sine_velocity(y_dist, aw)
        total_velocity = np.array((x_vel, y_vel))
        # heading direction
        heading = drone_to_ideal / 8 + total_velocity
        # return vector
        return heading / np.linalg.norm(heading)
    U, V = np.meshgrid(np.arange(-aw/2, aw/2 + diff, aw/(density - 1)),
                       np.arange(-al/2, al/2 + diff, al/(density - 1)))
    def saturate(mx, x):
        if x > mx:
            return mx
        elif x < -mx:
            return -mx
        else:
            return x
    for i in range(density):
        for o in range(density):
            pos = np.array((X[i][o], Y[i][o]))
            u, v = get_UV(pos)
            U[i][o] = saturate(60, u)
            V[i][o] = v
    color_array = np.sqrt(((V))**2)
    setup_plt()
    Q = plt.quiver(X, Y, U, V, color_array)
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    plt.savefig(media_path + "arrowplot.pdf", format='pdf')
    raise SystemExit(0)

# Adds a drone at a random position outside the avalance
def get_start_pos():
    def random_start():
        return np.array((np.random.randint(-aw / 2 - mld, aw / 2 + mld + 1),
                         np.random.randint(-al / 2 - mld, al / 2 + mld + 1)))
    start_pos = random_start()
    if not drones_start_in_avalanche:
        while ((-aw/2 < start_pos[0] < aw/2) and (-al/2 < start_pos[1] < al/2)):
            start_pos = random_start()
    return start_pos

# Set random times and positions if missing
if len(start_times) == 0:
    start_times.append(0)
while len(start_times) < no_drones:
    start_times.append(start_times[-1] + 60 + np.random.randint(-10, 11))
while len(start_positions) < no_drones:
    start_positions.append(get_start_pos())

# Add victims if not already defined
if no_victims > len(victim_pos):
    for i in range(no_victims - len(victim_pos)):
        victim_pos.append(np.array((
            np.random.randint(-aw/2, aw/2 + 1),
            np.random.randint(-al/2, al/2 + 1))))

# Run the simulation for the given number of time steps
for i in range(int(np.ceil(sim_time / sim_step))):
    time = sim_step * i
    for o in range(len(drone_objs)):
        drone_objs[o].update(sim_step)
        histories[str(o)].append(drone_objs[o].pos)
        if len(histories[str(o)]) > 1:
            if tuple(histories[str(o)][-1]) == tuple(histories[str(o)][-2]):
                end_sim += 1
    for o in range(len(start_times)):
        if start_times[o] == time:
            drone_objs.append(drone(start_positions[o], aw, al))
            histories[str(o)] = [start_positions[o]]
    if victim_found:
        print('Victim found after ' + "{:.1f}".format((sim_step * i)) + ' seconds')
        victim_found = False
    if end_sim == len(drone_objs):
        steps = i + 1
        print('simulation ending at step ' + str(i))
        print('simulation time = ' + "{:.1f}".format((sim_step * i)))
        break
    else:
        end_sim = 0

# Function to plot the data for a given step
def create_hist_plot(step, save):
    # Create final plot
    setup_plt()
    # Plot victims
    if no_victims > 0:
        x_victims = []
        y_victims = []
        for i in victim_pos:
            x_victims.append(i[0])
            y_victims.append(i[1])
        plt.scatter(x_victims, y_victims, marker='x', color='b', s=200)
    # Plot drone paths
    for i in histories:
        time = int(step - start_times[int(i)] / sim_step)
        if time > 0:
            x_hist = []
            y_hist = []
            for o in range(time):
                x_hist.append(histories[i][o][0])
                y_hist.append(histories[i][o][1])
            plt.plot(x_hist, y_hist)
            if show_swept_area:
                plt.plot(x_hist, y_hist, lw=2*beacon_range*(274/max_length),
                         alpha=0.12, zorder=-1, color='c', solid_capstyle='round')
            plt.scatter(x_hist[-1], y_hist[-1], marker='x', color='r')
    # Plot time
    if show_time:
        plt.text(max_length-5, 5-max_length, "Elapsed time (s): "+str(int((step)*sim_step)),
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    # Save plot
    plt.savefig(save)
    plt.show()

# Create GIF if requested or plot all data
if plot_gif:
    skip = int(gif_frame_speed / sim_step)
    count = 0
    for i in range(0, steps, skip):
        count = int(i / skip)
        create_hist_plot(i, gif_frame_storage+str(count)+'.png')
        images.append(imageio.imread(gif_frame_storage+str(count)+'.png'))
    for o in range(gif_end_freeze_time * 10):
        create_hist_plot(steps, gif_frame_storage+str(count + o + 1)+'.png')
        images.append(imageio.imread(gif_frame_storage+str(count + o + 1)+'.png'))
    imageio.mimsave(media_path + result_name+'.gif', images)
else:
    create_hist_plot(steps, media_path + result_name+'.png')

# Potential improvements for future
#   - in initalize only calculate nearest corner once?
#   - Remove self.branch, replace with self.sweepdir permanantly
#   - Drones > sweep area? FIX
#   - Arrow plots

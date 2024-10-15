import random
import time
import tkinter as tk
import threading

# Constants
COMMANDS = ['>', '<', '+', '-', '.', ',', '[', ']']
INITIAL_GRID_WIDTH = 30
INITIAL_GRID_HEIGHT = 6
WORLD_SIZE = 100  # Size of the 2D world
INITIAL_ENERGY = 200  # Initial energy for each organism
ENERGY_CONSUMPTION_RATE = 0.01  # Energy consumed per command
ENERGY_DECAY_RATE = 0.02  # Energy decay over time per step
ENERGY_GAIN_FROM_FOOD = 400  # Energy gained from consuming food
MUTATION_RATE = 0.05  # Probability of mutation per gene during reproduction
INACTIVITY_THRESHOLD = 50  # Number of steps without movement before mutation occurs
IDLE_MUTATION_RATE = 0.25  # Higher mutation rate due to inactivity
MAX_GRID_WIDTH = 50  # Maximum allowed grid width
MAX_GRID_HEIGHT = 50  # Maximum allowed grid height
REPRODUCTION_ENERGY_FACTOR = 0.9  # Energy cost per code unit for reproduction
SIMULATION_STEP_DELAY = 25  # Delay in milliseconds between simulation steps
FOOD_PERCENTAGE_THRESHOLD = 0.2  # Threshold for food percentage in the world
MAX_COMMANDS_PER_STEP = 1000  # Max number of Brainfuck commands executed per organism per step
MEMORY_ENERGY_COST = 0.01  # Energy cost per memory cell per step
MAX_MEMORY_CAPACITY = 1000  # Maximum number of memory cells per organism
CELL_11_MUTATION_RATE = 0.01  # Mutation rate for cell_11_value

# Movement command values stored in Cell 0
MOVEMENT_COMMANDS = {
    1: 'move_forward',
    2: 'move_backward',
    3: 'turn_left',
    4: 'turn_right',
    5: 'move_left',
    6: 'move_right'
}

# Breeding command values stored in Cell 2
BREEDING_COMMANDS = {
    1: 'mitosis',
    2: 'mutate_self',
    3: 'breed_with_neighbor'
}

# Fighting command values stored in Cell 3
FIGHTING_COMMANDS = {
    1: 'push',
    2: 'pull',
    3: 'attack'
}

class World:
    def __init__(self, size):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]  # Organisms' positions
        self.food_grid = [[False for _ in range(size)] for _ in range(size)]  # Grid to keep track of food positions
        self.organisms = []
        self.blink_counter = 0  # Counter to manage blinking effect

        # Set up the GUI
        self.root = tk.Tk()
        self.root.title("Organism World Simulation")

        # Create a Frame to hold the Canvas
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create the Canvas
        self.canvas = tk.Canvas(self.frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind the resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        self.cell_size = 20  # Initial cell size

        self.populate_food()

    def on_canvas_resize(self, event):
        # Recalculate cell size based on new canvas size
        self.canvas_width = event.width
        self.canvas_height = event.height
        # Calculate the new cell size to fit the grid within the canvas
        self.cell_size = min(self.canvas_width / self.size, self.canvas_height / self.size)
        # Redraw the grid
        self.display_world()

    def populate_food(self):
        """Ensure that FOOD_PERCENTAGE_THRESHOLD of the grid cells have food."""
        current_food_count = sum(sum(1 for cell in row if cell) for row in self.food_grid)
        required_food_count = int((self.size * self.size) * FOOD_PERCENTAGE_THRESHOLD)
        if current_food_count < required_food_count:
            num_food_items_to_add = required_food_count - current_food_count
            for _ in range(num_food_items_to_add):
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
                while self.food_grid[x][y]:  # Ensure we don't place food on an already occupied cell
                    x = random.randint(0, self.size - 1)
                    y = random.randint(0, self.size - 1)
                self.food_grid[x][y] = True

    def add_organism(self, organism, x, y):
        if self.grid[x][y] is None:
            self.grid[x][y] = organism
            organism.position = [x, y]
            self.organisms.append(organism)
        else:
            print(f"Position ({x}, {y}) is already occupied.")

    def display_world(self):
        self.canvas.delete("all")
        for x in range(self.size):
            for y in range(self.size):
                x1 = y * self.cell_size
                y1 = x * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if self.grid[x][y] is None:
                    color = "white"
                    if self.food_grid[x][y]:
                        color = "yellow"  # Represent food with yellow color
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=color, outline="black"
                    )
                else:
                    organism = self.grid[x][y]
                    # Determine color based on computing state and blink_counter
                    if organism.is_computing and self.blink_counter % 2 == 0:
                        color = "red"  # Blinking color when computing
                    else:
                        color = "green"  # Normal organism color
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=color, outline="black"
                    )
        self.root.update()

    def run(self):
        self.root.after(0, self.simulation_step)
        self.root.mainloop()

    def simulation_step(self):
        self.blink_counter += 1  # Increment blink counter for blinking effect
        # Remove dead organisms
        self.organisms = [org for org in self.organisms if org.energy > 0]
        if not self.organisms:
            # Respawn a new batch of organisms if all are dead
            for _ in range(20):
                grid = generate_grid(INITIAL_GRID_WIDTH, INITIAL_GRID_HEIGHT)
                organism = OrganismInterpreter(grid, self)
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                self.add_organism(organism, x, y)

        # Update positions and run organism steps
        for organism in self.organisms:
            organism.run_step()
        self.update_positions()
        self.populate_food()  # Regenerate food if below threshold
        self.display_world()
        # Schedule next simulation step
        self.root.after(SIMULATION_STEP_DELAY, self.simulation_step)

    def update_positions(self):
        # Ensure all organisms have positions within bounds
        for organism in self.organisms:
            x, y = organism.position
            x = max(0, min(self.size - 1, x))
            y = max(0, min(self.size - 1, y))
            organism.position = [x, y]
            # Check for food at the new position
            if self.food_grid[x][y]:
                organism.energy += ENERGY_GAIN_FROM_FOOD
                self.food_grid[x][y] = False  # Remove the food
                print(f"Organism at ({x}, {y}) ate food and gained energy.")

        # Update the world grid
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        for organism in self.organisms:
            x, y = organism.position
            if self.grid[x][y] is None:
                self.grid[x][y] = organism
            else:
                # Handle collision if necessary
                pass  # You may implement collision handling here

    def get_adjacent_cells(self, x, y):
        # Return valid adjacent cell coordinates
        adjacent = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    adjacent.append((nx, ny))
        return adjacent

    def reproduce(self, parent, offspring_grid, cell_11_value, position):
        # Create offspring
        offspring = OrganismInterpreter(offspring_grid, self, cell_11_value=cell_11_value)
        offspring.energy = INITIAL_ENERGY
        # Place offspring in the world
        x, y = position
        self.add_organism(offspring, x, y)
        print(f"Organism at ({parent.position}) reproduced at ({x}, {y}).")

    def breed(self, org1, org2):
        # Calculate reproduction energy based on average code size of parents
        avg_code_size = (org1.grid_size() + org2.grid_size()) / 2
        reproduction_energy = avg_code_size * REPRODUCTION_ENERGY_FACTOR
        if org1.energy >= reproduction_energy and org2.energy >= reproduction_energy:
            # Create offspring
            offspring_grid = self.combine_and_mutate_grids(org1.grid, org2.grid)
            # Inherit cell_11_value
            offspring_cell_11_value = org1.cell_11_value if random.random() < 0.5 else org2.cell_11_value
            # Apply mutation to cell_11_value
            if random.random() < CELL_11_MUTATION_RATE:
                offspring_cell_11_value = random.randint(0, 255)

            # Find an empty adjacent cell to place the offspring
            empty_cells = [cell for cell in self.get_adjacent_cells(org1.position[0], org1.position[1])
                           if self.grid[cell[0]][cell[1]] is None]
            if empty_cells:
                x, y = random.choice(empty_cells)
                self.reproduce(org1, offspring_grid, offspring_cell_11_value, (x, y))
                # Reduce parents' energy
                org1.energy -= reproduction_energy
                org2.energy -= reproduction_energy
                print(f"Organisms at ({org1.position}) and ({org2.position}) bred at ({x}, {y}).")
            else:
                print(f"No space to place offspring near ({org1.position}).")
        else:
            print(f"Organisms at ({org1.position}) and ({org2.position}) do not have enough energy to breed.")

    def combine_and_mutate_grids(self, grid1, grid2=None):
        """Combine the grids of two parents and apply mutations, including grid expansion."""
        if grid2 is None:
            grid2 = grid1  # For mitosis, use the same grid

        # Determine new grid dimensions
        max_height = max(len(grid1), len(grid2))
        max_width = max(len(grid1[0]), len(grid2[0]))

        # Initialize new grid
        new_grid = []

        for row_idx in range(max_height):
            new_row = []
            for col_idx in range(max_width):
                # Get commands from parents if available
                cmd1 = grid1[row_idx][col_idx] if row_idx < len(grid1) and col_idx < len(grid1[0]) else ' '
                cmd2 = grid2[row_idx][col_idx] if row_idx < len(grid2) and col_idx < len(grid2[0]) else ' '
                # Randomly choose command from one of the parents
                cmd = random.choice([cmd1, cmd2])
                # Apply mutation
                if random.random() < MUTATION_RATE:
                    cmd = random.choice(COMMANDS + [' '])
                new_row.append(cmd)
            new_grid.append(new_row)

        # Mutations that can increase grid size
        if random.random() < MUTATION_RATE:
            expand_direction = random.choice(['vertical', 'horizontal'])
            if expand_direction == 'vertical' and len(new_grid) < MAX_GRID_HEIGHT:
                # Add a new row at the end
                new_row = [random.choice(COMMANDS + [' ']) for _ in range(len(new_grid[0]))]
                new_grid.append(new_row)
            elif expand_direction == 'horizontal' and len(new_grid[0]) < MAX_GRID_WIDTH:
                # Add a new column to each row
                for row in new_grid:
                    row.append(random.choice(COMMANDS + [' ']))
        return new_grid

def generate_grid(width, height):
    """Generates a grid with random Brainfuck commands or blank spaces."""
    grid = []
    open_brackets = 0
    close_brackets = 0

    for _ in range(height):
        row = []
        for _ in range(width):
            if open_brackets > close_brackets and random.random() < 0.1:
                # Favor closing brackets when there are more open ones
                command = ']'
                close_brackets += 1
            elif random.random() < 0.1 and open_brackets < (width * height) // 2:
                # Randomly add opening brackets with some limit
                command = '['
                open_brackets += 1
            else:
                command = random.choice(COMMANDS + [' '])
            row.append(command)
        grid.append(row)

    return grid

class OrganismInterpreter:
    def __init__(self, grid, world, cell_11_value=None):
        self.grid = grid
        self.world = world  # Reference to the world
        self.data = {}  # Dynamic memory tape
        self.output = ""
        self.position = [0, 0]  # Starting position of the organism
        self.direction = 'UP'   # Initial direction
        self.current_cols = [0 for _ in self.grid]  # Cursors for each row
        self.pointers_per_row = [0 for _ in self.grid]  # Separate data pointers for each row
        self.energy = INITIAL_ENERGY  # Initial energy
        self.steps_since_movement = 0  # Steps without movement
        self.last_position = self.position.copy()  # Track last position
        self.is_computing = False  # Indicates if the organism is computing
        self.build_bracket_map()  # Initial build of the bracket map

        # Initialize cell_11_value
        if cell_11_value is None:
            self.cell_11_value = random.randint(0, 255)
        else:
            self.cell_11_value = cell_11_value
        self.data[11] = self.cell_11_value

        # Initialize other cells (4-9)
        self.data[4] = 0  # Health status or damage level
        self.data[5] = 0  # Defense level
        self.data[6] = 0  # Attack level
        self.data[7] = 0  # Speed
        self.data[8] = 0  # Age or generation count
        self.data[9] = 0  # Mutation rate

        # New additions
        self.temp_storage_per_row = [0 for _ in self.grid]  # Temporary storage for each row
        self.comma_toggle_state = ['read' for _ in self.grid]  # Toggle state for each row

    def grid_size(self):
        """Returns the total number of commands in the grid."""
        return sum(len(row) for row in self.grid)

    def build_bracket_map(self):
        """Precompute matching brackets in the grid for loops and fix any unmatched brackets."""
        self.bracket_maps = []
        for row_idx, row in enumerate(self.grid):
            temp_stack = []
            bracket_map = {}
            for col_idx, command in enumerate(row):
                if command == '[':
                    temp_stack.append(col_idx)
                elif command == ']':
                    if not temp_stack:
                        # Unmatched closing bracket; remove it
                        self.grid[row_idx][col_idx] = ' '
                    else:
                        start = temp_stack.pop()
                        bracket_map[start] = col_idx
                        bracket_map[col_idx] = start
            # Remove any unmatched opening brackets
            while temp_stack:
                col_idx = temp_stack.pop()
                self.grid[row_idx][col_idx] = ' '
            self.bracket_maps.append(bracket_map)

    def build_bracket_map_for_row(self, row_idx):
        """Rebuilds the bracket map for a specific row after code modification."""
        row = self.grid[row_idx]
        temp_stack = []
        bracket_map = {}
        for col_idx, command in enumerate(row):
            if command == '[':
                temp_stack.append(col_idx)
            elif command == ']':
                if not temp_stack:
                    # Unmatched closing bracket; remove it
                    self.grid[row_idx][col_idx] = ' '
                else:
                    start = temp_stack.pop()
                    bracket_map[start] = col_idx
                    bracket_map[col_idx] = start
        # Remove any unmatched opening brackets
        while temp_stack:
            col_idx = temp_stack.pop()
            self.grid[row_idx][col_idx] = ' '
        self.bracket_maps[row_idx] = bracket_map

    def update_sensory_cells(self):
        """Update cells 10, 12, and 13-20 with sensory data and energy state."""
        # Update cell 10
        x, y = self.position
        dx, dy = 0, 0
        if self.direction == 'UP':
            dx, dy = -1, 0
        elif self.direction == 'DOWN':
            dx, dy = 1, 0
        elif self.direction == 'LEFT':
            dx, dy = 0, -1
        elif self.direction == 'RIGHT':
            dx, dy = 0, 1

        look_x, look_y = x + dx, y + dy

        if not (0 <= look_x < self.world.size and 0 <= look_y < self.world.size):
            # Wall
            self.data[10] = 0
            self.data[12] = 0  # No organism ahead
        else:
            if self.world.food_grid[look_x][look_y]:
                # Energy (food)
                self.data[10] = 1
                self.data[12] = 0  # No organism ahead
            elif self.world.grid[look_x][look_y] is not None:
                # Another organism
                self.data[10] = 2
                # Update cell 12 with the other organism's cell_11_value
                other_organism = self.world.grid[look_x][look_y]
                self.data[12] = other_organism.cell_11_value
            else:
                # Empty space
                self.data[10] = 3  # Assign a value for empty space if desired
                self.data[12] = 0  # No organism ahead

        # Update cells 13-20 with energy state
        energy_int = int(abs(self.energy) * 1000)  # Use absolute value
        energy_str = str(energy_int).zfill(8)  # Ensure it has 8 digits

        # Optionally, store the sign in a separate cell if needed
        self.data[21] = 0 if self.energy >= 0 else 1  # 0 for positive, 1 for negative

        for i in range(8):
            self.data[13 + i] = int(energy_str[i])


    def run_step(self):
        """Run a single step of the organism's brain."""
        self.is_computing = True
        moved = False

        self.update_sensory_cells()  # Update cells 10, 12, 13-20

        commands_executed = 0
        max_commands_per_step = MAX_COMMANDS_PER_STEP

        while commands_executed < max_commands_per_step:
            for row_idx in range(len(self.grid)):
                if commands_executed >= max_commands_per_step:
                    break
                command = self.get_current_command(row_idx)
                if command:
                    jump_occurred = self.process_brain_command(row_idx, command)
                    if not jump_occurred:
                        self.advance_pointer(row_idx)
                    commands_executed += 1
                else:
                    continue  # No more commands in this row

        # Consume energy
        memory_cells_used = len(self.data)
        energy_loss = (commands_executed * ENERGY_CONSUMPTION_RATE +
                       ENERGY_DECAY_RATE +
                       memory_cells_used * MEMORY_ENERGY_COST)
        self.energy -= energy_loss

        if self.energy <= 0:
            self.energy = 0
            self.is_computing = False
            return  # Stop processing if energy is depleted


        # Check for inactivity
        if self.position == self.last_position:
            self.steps_since_movement += 1
            if self.steps_since_movement >= INACTIVITY_THRESHOLD:
                self.mutate_due_to_inactivity()
                self.steps_since_movement = 0  # Reset counter after mutation
        else:
            self.steps_since_movement = 0  # Reset counter if moved

        self.last_position = self.position.copy()
        self.is_computing = False

    def get_current_command(self, row_idx):
        row = self.grid[row_idx]
        col = self.current_cols[row_idx]
        if col < len(row):
            return row[col]
        return None

    def advance_pointer(self, row_idx):
        self.current_cols[row_idx] += 1
        if self.current_cols[row_idx] >= len(self.grid[row_idx]):
            self.current_cols[row_idx] = 0  # Loop back to start of the row

    def process_brain_command(self, row_idx, command):
        """Processes a single Brainfuck command for a specific row."""
        jump_occurred = False
        pointer = self.pointers_per_row[row_idx]
        bracket_map = self.bracket_maps[row_idx]
        current_col = self.current_cols[row_idx]

        if command == '>':
            pointer += 1
        elif command == '<':
            pointer -= 1
            if pointer < 0:
                pointer = 0  # Prevent negative pointers

        # Enforce memory capacity limit
        if len(self.data) >= MAX_MEMORY_CAPACITY and pointer not in self.data:
            # Skip allocating new memory cell
            pass

        # Access or initialize the memory cell
        cell_value = self.data.get(pointer, 0)

        if command == '+':
            cell_value = (cell_value + 1) % 256
            self.data[pointer] = cell_value
            if cell_value == 0 and pointer in self.data:
                del self.data[pointer]  # Deallocate memory if zero
        elif command == '-':
            cell_value = (cell_value - 1) % 256
            self.data[pointer] = cell_value
            if cell_value == 0 and pointer in self.data:
                del self.data[pointer]  # Deallocate memory if zero
        elif command == '.':
            self.execute_cell_action(pointer)
        elif command == ',':
            # Modified comma command handling
            if self.comma_toggle_state[row_idx] == 'read':
                # Read sensory input from the environment
                distance = self.get_distance_to_food()
                self.temp_storage_per_row[row_idx] = distance
                print(f"Row {row_idx}: Read value {distance} into temporary storage.")
                # Toggle to write mode
                self.comma_toggle_state[row_idx] = 'write'
            else:
                # Write from temporary storage to data tape
                self.data[pointer] = self.temp_storage_per_row[row_idx]
                print(f"Row {row_idx}: Wrote value {self.temp_storage_per_row[row_idx]} from temporary storage to data cell {pointer}.")
                # Toggle back to read mode
                self.comma_toggle_state[row_idx] = 'read'
        elif command == '[':
            if cell_value == 0:
                if current_col in bracket_map:
                    self.current_cols[row_idx] = bracket_map[current_col]
                    jump_occurred = True
        elif command == ']':
            if cell_value != 0:
                if current_col in bracket_map:
                    self.current_cols[row_idx] = bracket_map[current_col]
                    jump_occurred = True

        self.pointers_per_row[row_idx] = pointer
        return jump_occurred

    def execute_cell_action(self, pointer):
        """Executes the action associated with the current data cell."""
        cell_value = self.data.get(pointer, 0)

        if pointer == 0:
            # Movement commands
            if cell_value in MOVEMENT_COMMANDS:
                action = MOVEMENT_COMMANDS[cell_value]
                getattr(self, action)()
                # Reset the movement command indicator
                self.data[pointer] = 0
        elif pointer == 1:
            # Direct code editing
            self.direct_code_edit(self.pointers_per_row.index(pointer))
        elif pointer == 2:
            # Breeding commands
            if cell_value in BREEDING_COMMANDS:
                action = BREEDING_COMMANDS[cell_value]
                getattr(self, action)()
                # Reset the breeding command indicator
                self.data[pointer] = 0
        elif pointer == 3:
            # Fighting commands
            if cell_value in FIGHTING_COMMANDS:
                action = FIGHTING_COMMANDS[cell_value]
                getattr(self, action)()
                # Reset the fighting command indicator
                self.data[pointer] = 0
        else:
            # For other cells, you may define additional behaviors or do nothing
            pass

    def direct_code_edit(self, row_idx):
        """Allows direct editing of the code in the current row."""
        # Get the value from cell 1
        edit_value = self.data.get(1, 0)
        # Map the edit_value to a command or a space
        possible_commands = COMMANDS + [' ']
        command_index = edit_value % len(possible_commands)
        new_command = possible_commands[command_index]

        # Choose a random position in the current row to modify
        row_length = len(self.grid[row_idx])
        if row_length == 0:
            return  # Nothing to edit in an empty row

        edit_position = random.randint(0, row_length - 1)
        old_command = self.grid[row_idx][edit_position]
        self.grid[row_idx][edit_position] = new_command

        print(f"Organism at {self.position} edited row {row_idx}, position {edit_position}: '{old_command}' -> '{new_command}'")

        # Rebuild the bracket map for the modified row
        self.build_bracket_map_for_row(row_idx)

    def get_distance_to_food(self):
        """Calculates the distance to the nearest food in the direction the organism is facing."""
        x, y = self.position
        max_distance = self.world.size  # Maximum possible distance
        distance = 1

        dx, dy = 0, 0
        if self.direction == 'UP':
            dx, dy = -1, 0
        elif self.direction == 'DOWN':
            dx, dy = 1, 0
        elif self.direction == 'LEFT':
            dx, dy = 0, -1
        elif self.direction == 'RIGHT':
            dx, dy = 0, 1

        while 0 <= x + dx * distance < self.world.size and 0 <= y + dy * distance < self.world.size:
            nx, ny = x + dx * distance, y + dy * distance
            if self.world.food_grid[nx][ny]:
                return distance
            distance += 1
            if distance >= max_distance:
                break

        return 0  # Return 0 if no food is found in the direction

    def mutate_due_to_inactivity(self):
        """Apply random mutations to the code due to inactivity."""
        print(f"Organism at {self.position} is inactive. Applying mutations to its code.")
        for row_idx in range(len(self.grid)):
            for col_idx in range(len(self.grid[row_idx])):
                if random.random() < IDLE_MUTATION_RATE:
                    self.grid[row_idx][col_idx] = random.choice(COMMANDS + [' '])
        self.build_bracket_map()  # Rebuild bracket map after mutation

    # Movement methods with boundary checks
    def turn_left(self):
        """Turn the organism left."""
        directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        idx = directions.index(self.direction)
        self.direction = directions[(idx + 1) % 4]
        print(f"Organism at {self.position} turned left, now facing {self.direction}")

    def turn_right(self):
        """Turn the organism right."""
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.direction)
        self.direction = directions[(idx + 1) % 4]
        print(f"Organism at {self.position} turned right, now facing {self.direction}")

    def move_forward(self):
        """Propel the organism forward based on the current direction."""
        dx, dy = self.get_direction_delta()
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        if 0 <= new_x < self.world.size and 0 <= new_y < self.world.size and self.world.grid[new_x][new_y] is None:
            self.position[0] = new_x
            self.position[1] = new_y
            print(f"Organism moved forward to position {self.position}")
        else:
            print(f"Organism at {self.position} cannot move forward.")

    def move_backward(self):
        """Propel the organism backward based on the current direction."""
        dx, dy = self.get_direction_delta()
        new_x = self.position[0] - dx
        new_y = self.position[1] - dy
        if 0 <= new_x < self.world.size and 0 <= new_y < self.world.size and self.world.grid[new_x][new_y] is None:
            self.position[0] = new_x
            self.position[1] = new_y
            print(f"Organism moved backward to position {self.position}")
        else:
            print(f"Organism at {self.position} cannot move backward.")

    def move_left(self):
        """Propel the organism left (relative movement)."""
        new_x = self.position[0]
        new_y = self.position[1] - 1
        if 0 <= new_y < self.world.size and self.world.grid[new_x][new_y] is None:
            self.position[1] = new_y
            print(f"Organism moved left to position {self.position}")
        else:
            print(f"Organism at {self.position} cannot move left.")

    def move_right(self):
        """Propel the organism right (relative movement)."""
        new_x = self.position[0]
        new_y = self.position[1] + 1
        if 0 <= new_y < self.world.size and self.world.grid[new_x][new_y] is None:
            self.position[1] = new_y
            print(f"Organism moved right to position {self.position}")
        else:
            print(f"Organism at {self.position} cannot move right.")

    # Breeding methods
    def mitosis(self):
        """Reproduce asexually by creating a clone with possible mutations."""
        # Check if the organism has enough energy
        reproduction_energy = self.grid_size() * REPRODUCTION_ENERGY_FACTOR
        if self.energy >= reproduction_energy:
            # Create offspring grid with mutations
            offspring_grid = self.world.combine_and_mutate_grids(self.grid)
            # Mutate cell_11_value if necessary
            offspring_cell_11_value = self.cell_11_value
            if random.random() < CELL_11_MUTATION_RATE:
                offspring_cell_11_value = random.randint(0, 255)
            # Find an empty adjacent cell
            empty_cells = [cell for cell in self.world.get_adjacent_cells(self.position[0], self.position[1])
                           if self.world.grid[cell[0]][cell[1]] is None]
            if empty_cells:
                x, y = random.choice(empty_cells)
                self.world.reproduce(self, offspring_grid, offspring_cell_11_value, (x, y))
                self.energy -= reproduction_energy
            else:
                print(f"No space to place offspring near ({self.position}).")
        else:
            print(f"Organism at {self.position} does not have enough energy for mitosis.")

    def mutate_self(self):
        """Cause self-mutation in the organism's code."""
        print(f"Organism at {self.position} is mutating itself.")
        for row_idx in range(len(self.grid)):
            for col_idx in range(len(self.grid[row_idx])):
                if random.random() < MUTATION_RATE:
                    self.grid[row_idx][col_idx] = random.choice(COMMANDS + [' '])
        self.build_bracket_map()

    def breed_with_neighbor(self):
        """Attempt to breed with an adjacent organism."""
        x, y = self.position
        # Check adjacent cells for other organisms
        neighbors = self.world.get_adjacent_cells(x, y)
        for nx, ny in neighbors:
            neighbor_org = self.world.grid[nx][ny]
            if neighbor_org and neighbor_org != self:
                # Attempt to breed
                self.world.breed(self, neighbor_org)
                return  # Breed with only one neighbor

    # Fighting methods
    def push(self):
        """Push the organism in front backward."""
        target = self.get_organism_in_front()
        if target:
            dx, dy = self.get_direction_delta()
            new_x = target.position[0] + dx
            new_y = target.position[1] + dy
            # Check if the new position is within bounds and not occupied
            if 0 <= new_x < self.world.size and 0 <= new_y < self.world.size and self.world.grid[new_x][new_y] is None:
                target.position[0] = new_x
                target.position[1] = new_y
                print(f"Organism at {self.position} pushed organism to {target.position}")
            else:
                print(f"Cannot push organism at {target.position} to out of bounds or occupied cell.")
        else:
            print(f"No organism to push in front of {self.position}")

    def pull(self):
        """Pull the organism in front towards self."""
        target = self.get_organism_in_front()
        if target:
            dx, dy = self.get_direction_delta()
            new_x = target.position[0] - dx
            new_y = target.position[1] - dy
            # Check if the new position is within bounds and not occupied
            if 0 <= new_x < self.world.size and 0 <= new_y < self.world.size and self.world.grid[new_x][new_y] is None:
                target.position[0] = new_x
                target.position[1] = new_y
                print(f"Organism at {self.position} pulled organism to {target.position}")
            else:
                print(f"Cannot pull organism at {target.position} to out of bounds or occupied cell.")
        else:
            print(f"No organism to pull in front of {self.position}")

    def attack(self):
        """Attack the organism in front, draining its energy."""
        target = self.get_organism_in_front()
        if target:
            damage = 50  # Define damage value
            target.energy -= damage
            self.energy += damage * 0.5  # Absorb some energy from the target
            print(f"Organism at {self.position} attacked organism at {target.position}")
        else:
            print(f"No organism to attack in front of {self.position}")

    def get_organism_in_front(self):
        """Get the organism in the cell in front of the current organism."""
        x, y = self.position
        dx, dy = self.get_direction_delta()
        look_x, look_y = x + dx, y + dy
        if 0 <= look_x < self.world.size and 0 <= look_y < self.world.size:
            return self.world.grid[look_x][look_y]
        return None

    def get_direction_delta(self):
        """Get the delta for the current direction."""
        if self.direction == 'UP':
            return -1, 0
        elif self.direction == 'DOWN':
            return 1, 0
        elif self.direction == 'LEFT':
            return 0, -1
        elif self.direction == 'RIGHT':
            return 0, 1
        return 0, 0  # Default

# Create the world
world = World(WORLD_SIZE)

# Generate organisms and add them to the world
for _ in range(20):
    grid = generate_grid(INITIAL_GRID_WIDTH, INITIAL_GRID_HEIGHT)
    organism = OrganismInterpreter(grid, world)
    x, y = random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1)
    world.add_organism(organism, x, y)

# Run the world simulation
print("Initial World State:")
world.display_world()
world.run()

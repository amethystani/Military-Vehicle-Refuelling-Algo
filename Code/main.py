from moviepy.video.io.bindings import mplfig_to_npimage
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from moviepy.editor import  VideoClip
import os
import csv
import time
from contextlib import contextmanager
import datetime

random.seed(None)  # This ensures a different random seed each time
np.random.seed (None)
# Constants
MAP_SIZE = 40
DAYS = 15
BASE_FUEL_THRESHOLD = 0.25
REFUEL_AMOUNT = 0.70
STRANDED_FUEL_LEVEL = 10
PATROL_DURATION = 100
alerts = []
MAX_ALERTS = 5
alert_queue = deque(maxlen=MAX_ALERTS)

#dict
vehicles = [
    {"id": 1, "fuel_capacity": 100, "consumption_rate": 0.5, "position": np.array([10, 10]), "fuel_level": 100,
     "path": [[10, 10]], "patrol_area": (0, 0, 20, 20), "state": "patrolling"},
    {"id": 2, "fuel_capacity": 100, "consumption_rate": 0.5, "position": np.array([30, 10]), "fuel_level": 100,
     "path": [[30, 10]], "patrol_area": (20, 0, 40, 20), "state": "patrolling"},
    {"id": 3, "fuel_capacity": 100, "consumption_rate": 0.5, "position": np.array([10, 30]), "fuel_level": 100,
     "path": [[10, 30]], "patrol_area": (0, 20, 20, 40), "state": "patrolling"},
    {"id": 4, "fuel_capacity": 100, "consumption_rate": 0.5, "position": np.array([30, 30]), "fuel_level": 100,
     "path": [[30, 30]], "patrol_area": (20, 20, 40, 40), "state": "patrolling"}
]

def ensure_videos_directory():
    if not os.path.exists("Videos"):
        os.makedirs("Videos")

def generate_random_hub_positions():
    hub_positions = []
    quadrants = [(0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 40, 40)]  # 3quads

    for quadrant in quadrants:
        x_min, y_min, x_max, y_max = quadrant

        # Decide randomly whether to place the hub in the center or corner
        if random.choice([True, False]):
            # Place in center
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
        else:
            # Place in corner
            x = random.choice([x_min, x_max])
            y = random.choice([y_min, y_max])

        hub_positions.append(np.array([x, y]))

    return hub_positions

# Generate random hub positions
random_hub_positions = generate_random_hub_positions()
# Generate random hub positions for all 15 days
all_hub_positions = [generate_random_hub_positions() for _ in range(DAYS)]


# Define hubs with random positions and additional fees
hubs = [
    {"id": 1, "position": all_hub_positions[0][0], "fee": 1.00},
    {"id": 2, "position": all_hub_positions[0][1], "fee": 1.10},
    {"id": 3, "position": all_hub_positions[0][2], "fee": 1.10}
]

# Refuel vehicles (one for each hub)
refuel_vehicles = [
    {"id": i + 1, "position": hub["position"].copy(), "path": [], "state": "idle", "target": None, "speed": 1,
     "hub": hub, "refuel_point": None}
    for i, hub in enumerate(hubs)
]

# Create graph for pathfinding
G = nx.grid_2d_graph(MAP_SIZE + 1, MAP_SIZE + 1)

# Genetic Algorithm parameters
POPULATION_SIZE = 10
GENERATIONS = 1000
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8


def shortest_path(start, end):
    start = tuple(map(int, start))
    end = tuple(map(int, end))
    return nx.shortest_path(G, source=start, target=end)


def calculate_refuel_cost(hub, fuel_amount):
    return hub["fee"] * fuel_amount
def find_nearest_hub(position):
    return min(hubs, key=lambda h: np.linalg.norm(h["position"] - position))
def calculate_dynamic_threshold(vehicle, nearest_hub):
    distance_to_hub = np.linalg.norm(vehicle["position"] - nearest_hub["position"])
    return BASE_FUEL_THRESHOLD + (distance_to_hub / MAP_SIZE) * 0.2
def predict_refuel_need(vehicle, patrol_duration):
    predicted_consumption = vehicle["consumption_rate"] * patrol_duration
    return vehicle["fuel_level"] - predicted_consumption <= BASE_FUEL_THRESHOLD * vehicle["fuel_capacity"]
def patrol(vehicle):
    if vehicle["state"] == "waiting_for_refuel":
        patrol_area = vehicle["safe_patrol_area"]
    else:
        patrol_area = vehicle["patrol_area"]

    direction = np.random.choice(['x', 'y'])
    step = np.random.choice([1, 2])
    movement = np.zeros(2)
    movement[0 if direction == 'x' else 1] = np.random.choice([-step, step])

    new_position = np.clip(
        vehicle["position"] + movement,
        [patrol_area[0] + 1, patrol_area[1] + 1],
        [patrol_area[2] - 1, patrol_area[3] - 1]
    )

    distance_traveled = np.linalg.norm(new_position - vehicle["position"])
    vehicle["position"] = new_position
    vehicle["path"].append(new_position.tolist())
    return distance_traveled
def calculate_daily_refueling_cost(vehicles, refuel_vehicles, hubs):
    total_cost = 0

    for vehicle in vehicles:
        # Cost for movement
        total_cost += len(vehicle["path"]) * vehicle["consumption_rate"] * 10

        # Cost for refueling
        if vehicle["state"] == "refueling":
            nearest_hub = find_nearest_hub(vehicle["position"])
            refuel_amount = vehicle["fuel_capacity"] - vehicle["fuel_level"]
            total_cost += calculate_refuel_cost(nearest_hub, refuel_amount) * 100

        # Operational cost per vehicle per day
        total_cost += 1000

    for refuel_vehicle in refuel_vehicles:
        # Cost for refuel vehicle movement
        total_cost += len(refuel_vehicle["path"]) * 20

        # Operational cost per refuel vehicle per day
        total_cost += 2000

    # Add penalties for stranded vehicles
    stranded_vehicles = sum(1 for v in vehicles if v["state"] == "stranded")
    total_cost += stranded_vehicles * 10000

    # Add cost for maintaining hubs
    total_cost += len(hubs) * 5000

    return total_cost
def create_individual():
    return {
        "dispatch_strategy": [random.choice(hubs)["id"] for _ in range(len(vehicles))],
        "gather_strategy": [random.choice([True, False]) for _ in range(len(vehicles))],
        "refuel_threshold": random.uniform(0.2, 0.5),
        "emergency_threshold": random.uniform(0.1, 0.3)
    }
def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

import logging
logging.basicConfig(filename='fitness_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def fitness(individual):
    total_cost = 0
    total_distance = 0
    emergency_refuels = 0
    stranded_vehicles = 0

    for i, vehicle in enumerate(vehicles):
        hub_id = individual["dispatch_strategy"][i]
        hub = next(h for h in hubs if h["id"] == hub_id)
        gather = individual["gather_strategy"][i]

        nearest_hub = find_nearest_hub(vehicle["position"])
        dynamic_threshold = calculate_dynamic_threshold(vehicle, nearest_hub)

        if vehicle["fuel_level"] <= dynamic_threshold * vehicle["fuel_capacity"]:
            if gather:
                gathering_point = np.mean([v["position"] for v in vehicles], axis=0)
                distance_to_gather = np.linalg.norm(vehicle["position"] - gathering_point)
            else:
                distance_to_gather = np.linalg.norm(vehicle["position"] - hub["position"])

            total_distance += distance_to_gather
            total_cost += distance_to_gather * vehicle["consumption_rate"] * 1.5

            refuel_amount = vehicle["fuel_capacity"] - vehicle["fuel_level"]
            total_cost += calculate_refuel_cost(hub, refuel_amount)

        if vehicle["fuel_level"] <= individual["emergency_threshold"] * vehicle["fuel_capacity"]:
            emergency_refuels += 1

        if vehicle["fuel_level"] <= 0:
            stranded_vehicles += 1

    avg_distance_to_hub = np.mean([
        np.linalg.norm(
            v["position"] - next(h["position"] for h in hubs if h["id"] == individual["dispatch_strategy"][i]))
        for i, v in enumerate(vehicles)
    ])

    hub_usage = {hub["id"]: 0 for hub in hubs}
    for hub_id in individual["dispatch_strategy"]:
        hub_usage[hub_id] += 1
    hub_balance = max(hub_usage.values()) - min(hub_usage.values())

    # Normalize the values
    max_cost = 1000000  # Adjust based on your expected maximum cost
    max_distance = MAP_SIZE * 2  # Maximum possible distance
    max_emergency = len(vehicles)  # Maximum possible emergency refuels
    max_hub_balance = len(vehicles)  # Maximum possible hub imbalance

    normalized_cost = 1 - (total_cost / max_cost)
    normalized_distance = 1 - (total_distance / max_distance)
    normalized_stranded = 1 - (stranded_vehicles / len(vehicles))
    normalized_emergency = 1 - (emergency_refuels / max_emergency)
    normalized_avg_distance = 1 - (avg_distance_to_hub / max_distance)
    normalized_hub_balance = 1 - (hub_balance / max_hub_balance)

    # Weighted sum (adjust weights as needed)
    fitness_score = (
        0.3 * normalized_cost +
        0.2 * normalized_distance +
        0.2 * normalized_stranded +
        0.1 * normalized_emergency +
        0.1 * normalized_avg_distance +
        0.1 * normalized_hub_balance
    )


    # Log the fitness score and its components
    logging.info(f"Fitness: {fitness_score:.6f}")

    return fitness_score


def select_parents(population):
    return random.choices(
        population,
        weights=[fitness(ind) for ind in population],
        k=2
    )
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(vehicles) - 1)
        child = {
            "dispatch_strategy": parent1["dispatch_strategy"][:crossover_point] + parent2["dispatch_strategy"][crossover_point:],
            "gather_strategy": parent1["gather_strategy"][:crossover_point] + parent2["gather_strategy"][crossover_point:],
            "refuel_threshold": (parent1["refuel_threshold"] + parent2["refuel_threshold"]) / 2,
            "emergency_threshold": (parent1["emergency_threshold"] + parent2["emergency_threshold"]) / 2
        }
        return child
    else:
        return random.choice([parent1, parent2])
def mutate(individual):
    for i in range(len(vehicles)):
        if random.random() < MUTATION_RATE:
            individual["dispatch_strategy"][i] = random.choice(hubs)["id"]
        if random.random() < MUTATION_RATE:
            individual["gather_strategy"][i] = not individual["gather_strategy"][i]

    if random.random() < MUTATION_RATE:
        individual["refuel_threshold"] = random.uniform(0.2, 0.5)
    if random.random() < MUTATION_RATE:
        individual["emergency_threshold"] = random.uniform(0.1, 0.3)

    return individual
def genetic_algorithm(progress_callback=None):
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8

    population = [create_individual() for _ in range(population_size)]
    best_fitness = float('-inf')
    best_individual = None

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [fitness(ind) for ind in population]

        # Select parents
        parents = random.choices(population, weights=fitnesses, k=population_size)

        # Create new population
        new_population = []
        for i in range(0, population_size, 2):
            child1 = crossover(parents[i], parents[i+1])
            child2 = crossover(parents[i], parents[i+1])

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        # Elitism: keep the best individual
        best_of_generation = max(population, key=fitness)
        new_population[0] = best_of_generation

        population = new_population

        generation_best_fitness = fitness(best_of_generation)
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = best_of_generation

        if progress_callback:
            progress_callback(f"Generation {generation + 1}: Best fitness = {best_fitness}")

    return best_individual, best_fitness
def apply_refueling_strategy(strategy):
    vehicles_needing_refuel = []
    for vehicle in vehicles:
        nearest_hub = find_nearest_hub(vehicle["position"])
        dynamic_threshold = calculate_dynamic_threshold(vehicle, nearest_hub)
        if (vehicle["state"] == "stranded" or
                vehicle["fuel_level"] <= dynamic_threshold * vehicle["fuel_capacity"] or
                predict_refuel_need(vehicle, PATROL_DURATION)):
            vehicles_needing_refuel.append(vehicle)

    if vehicles_needing_refuel:
        # Calculate the centroid of vehicles needing refuel
        centroid = np.mean([v["position"] for v in vehicles_needing_refuel], axis=0)

        # Sort vehicles by distance to centroid
        vehicles_needing_refuel.sort(key=lambda v: np.linalg.norm(v["position"] - centroid))

        # Find the closest idle refuel vehicle to the centroid
        idle_refuel_vehicles = [rv for rv in refuel_vehicles if rv["state"] == "idle"]
        if idle_refuel_vehicles:
            closest_refuel_vehicle = min(
                idle_refuel_vehicles,
                key=lambda rv: np.linalg.norm(rv["position"] - centroid)
            )

            vehicles_to_refuel = vehicles_needing_refuel[:2]  # Limit to 2 vehicles
            remaining_vehicles = vehicles_needing_refuel[2:]  # Remaining vehicles

            if vehicles_to_refuel:
                # Only create gathering point if there are vehicles to refuel
                gathering_point = np.mean([closest_refuel_vehicle["position"]] +
                                          [v["position"] for v in vehicles_to_refuel], axis=0)

                refuel_vehicle_time = len(shortest_path(closest_refuel_vehicle["position"], gathering_point))

                for vehicle in vehicles_to_refuel:
                    vehicle_time = len(shortest_path(vehicle["position"], gathering_point))
                    delay = max(0, refuel_vehicle_time - vehicle_time + 5)
                    vehicle["path"] = [vehicle["position"]] * delay + shortest_path(vehicle["position"], gathering_point)
                    vehicle["state"] = "moving_to_refuel"

                closest_refuel_vehicle["state"] = "moving_to_gather"
                closest_refuel_vehicle["target"] = vehicles_to_refuel
                closest_refuel_vehicle["refuel_point"] = gathering_point
                closest_refuel_vehicle["path"] = shortest_path(closest_refuel_vehicle["position"], gathering_point)

                # Clear refuel points for all other refuel vehicles
                for rv in refuel_vehicles:
                    if rv != closest_refuel_vehicle:
                        rv["refuel_point"] = None
                        rv["state"] = "idle"
                        rv["target"] = None
                        rv["path"] = []

            else:
                # If no vehicles to refuel, reset all refuel vehicles
                for rv in refuel_vehicles:
                    rv["state"] = "idle"
                    rv["target"] = None
                    rv["refuel_point"] = None
                    rv["path"] = []

        else:
            remaining_vehicles = vehicles_needing_refuel

        # Handle remaining vehicles
        for vehicle in remaining_vehicles:
            nearest_hub = find_nearest_hub(vehicle["position"])
            hub_distance = np.linalg.norm(vehicle["position"] - nearest_hub["position"])
            fuel_needed = vehicle["fuel_capacity"] - vehicle["fuel_level"]
            fuel_to_hub = vehicle["consumption_rate"] * hub_distance

            if vehicle["fuel_level"] > fuel_to_hub:
                # Option 1: Move to the nearest hub
                vehicle["path"] = shortest_path(vehicle["position"], nearest_hub["position"])
                vehicle["state"] = "moving_to_hub"
            else:
                # Option 2: Wait for the next refuel cycle
                vehicle["state"] = "waiting_for_refuel"
                # Calculate a safe patrolling area
                safe_radius = (vehicle["fuel_level"] / vehicle["consumption_rate"]) * 0.4  # 40% of max range
                vehicle["safe_patrol_area"] = (
                    max(vehicle["patrol_area"][0], vehicle["position"][0] - safe_radius),
                    max(vehicle["patrol_area"][1], vehicle["position"][1] - safe_radius),
                    min(vehicle["patrol_area"][2], vehicle["position"][0] + safe_radius),
                    min(vehicle["patrol_area"][3], vehicle["position"][1] + safe_radius)
                )

    else:
        # If no vehicles need refueling, reset all refuel vehicles
        for rv in refuel_vehicles:
            rv["state"] = "idle"
            rv["target"] = None
            rv["refuel_point"] = None
            rv["path"] = []
def add_alert(message):
    alert_queue.append(message)
def get_alerts():
    return list(alert_queue)
def extract_frame_data(frame, day, vehicles, refuel_vehicles, hubs):
    data = []
    for vehicle in vehicles:
        data.append({
            'frame': frame,
            'day': day,
            'entity_type': 'vehicle',
            'id': vehicle['id'],
            'x_position': vehicle['position'][0],
            'y_position': vehicle['position'][1],
            'fuel_level': vehicle['fuel_level'],
            'state': vehicle['state']
        })

    for rv in refuel_vehicles:
        data.append({
            'frame': frame,
            'day': day,
            'entity_type': 'refuel_vehicle',
            'id': rv['id'],
            'x_position': rv['position'][0],
            'y_position': rv['position'][1],
            'state': rv['state'],
            'refuel_point_x': rv['refuel_point'][0] if rv['refuel_point'] is not None else None,
            'refuel_point_y': rv['refuel_point'][1] if rv['refuel_point'] is not None else None
        })

    for hub in hubs:
        data.append({
            'frame': frame,
            'day': day,
            'entity_type': 'hub',
            'id': hub['id'],
            'x_position': hub['position'][0],
            'y_position': hub['position'][1]
        })

    return data
@contextmanager
def open_with_retry(filename, mode='r', max_attempts=5, delay=1):
    for attempt in range(max_attempts):
        try:
            with open(filename, mode) as f:
                yield f
            break
        except PermissionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)
def update_csv(data, day, csv_dir='Simulation_Data'):
    # Ensure the directory exists
    os.makedirs(csv_dir, exist_ok=True)

    csv_file = os.path.join(csv_dir, f'simulation_data_day_{day}.csv')
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'day', 'entity_type', 'id', 'x_position', 'y_position',
            'fuel_level', 'state', 'refuel_point_x', 'refuel_point_y'
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerows(data)
def update(frame, day):
    global vehicles, refuel_vehicles, current_day
    fig.clear()
    # Create two subplots - one for dashboard, one for map
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 9])  # Adjust ratio to make dashboard smaller
    ax_dashboard = fig.add_subplot(gs[0], frame_on=False)  # No frame for dashboard
    ax_map = fig.add_subplot(gs[1])

    if frame % 100 == 0:
        apply_refueling_strategy(best_strategy)

    for vehicle in vehicles:
        if vehicle["fuel_level"] <= 0:
            vehicle["state"] = "stranded"
            add_alert(f"Vehicle {vehicle['id']} has run out of fuel!")
            continue

        if vehicle["state"] == "patrolling":
            distance_traveled = patrol(vehicle)
            vehicle["fuel_level"] = max(0, vehicle["fuel_level"] - vehicle["consumption_rate"] * distance_traveled)
            if vehicle["fuel_level"] <= 30:
                add_alert(f"Vehicle {vehicle['id']} is low on fuel ({vehicle['fuel_level']:.1f})")
        elif vehicle["state"] in ["moving_to_refuel", "moving_to_hub"]:
            if vehicle["path"]:
                next_position = np.array(vehicle["path"][0])
                distance_to_next = np.linalg.norm(vehicle["position"] - next_position)
                if vehicle["fuel_level"] >= vehicle["consumption_rate"] * distance_to_next:
                    vehicle["position"] = next_position
                    vehicle["path"].pop(0)
                    vehicle["fuel_level"] -= vehicle["consumption_rate"] * distance_to_next
                else:
                    vehicle["state"] = "stranded"
                    add_alert(f"Vehicle {vehicle['id']} has run out of fuel while moving!")
            else:
                vehicle["state"] = "refueling"
                add_alert(f"Vehicle {vehicle['id']} has arrived for refueling")
        elif vehicle["state"] == "refueling":
            vehicle["fuel_level"] = vehicle["fuel_capacity"]
            vehicle["state"] = "patrolling"
            vehicle["path"] = [vehicle["position"].tolist()]
            add_alert(f"Vehicle {vehicle['id']} has been refueled")

    for refuel_vehicle in refuel_vehicles:
        if refuel_vehicle["state"] == "moving_to_gather":
            if refuel_vehicle["path"]:
                refuel_vehicle["position"] = np.array(refuel_vehicle["path"].pop(0))
                if not refuel_vehicle["path"]:
                    add_alert(f"Refuel vehicle from Hub {refuel_vehicle['hub']['id']} has arrived at gathering point")
            else:
                refuel_vehicle["state"] = "refueling"
        elif refuel_vehicle["state"] == "refueling":
            if all(v["state"] != "moving_to_refuel" for v in refuel_vehicle["target"]):
                refuel_vehicle["state"] = "returning_to_hub"
                refuel_vehicle["path"] = shortest_path(refuel_vehicle["position"], refuel_vehicle["hub"]["position"])
                refuel_vehicle["refuel_point"] = None
                add_alert(f"Refuel vehicle from Hub {refuel_vehicle['hub']['id']} is returning to hub")
        elif refuel_vehicle["state"] == "returning_to_hub":
            if refuel_vehicle["path"]:
                refuel_vehicle["position"] = np.array(refuel_vehicle["path"].pop(0))
            else:
                refuel_vehicle["state"] = "idle"
                refuel_vehicle["target"] = None
                refuel_vehicle["refuel_point"] = None
                add_alert(f"Refuel vehicle has returned to Hub {refuel_vehicle['hub']['id']}")
        elif refuel_vehicle["state"] == "idle":
            refuel_vehicle["target"] = None
            refuel_vehicle["refuel_point"] = None
            refuel_vehicle["path"] = []

    plot_entities(ax_map, vehicles, refuel_vehicles, hubs)
    ax_map.set_xlim(-1, MAP_SIZE + 1)
    ax_map.set_ylim(-1, MAP_SIZE + 1)
    # Create the title
    title = f'Vehicle Patrol and Refueling Simulation - Day {current_day}, Frame {frame + 1}'
    fig.suptitle(title, fontsize=10, y=0.98)
    # Draw the refuel dashboard on the left side of the dashboard area
    draw_refuel_dashboard(ax_dashboard, vehicles)
    # Display alerts in a box on the right side of the dashboard area
    alerts = get_alerts()
    alert_text = "\n".join(alerts[-4:]) if alerts else "No current alerts"  # Increased to show 4 alerts
    ax_dashboard.text(0.65, 0.5, alert_text,
                      transform=ax_dashboard.transAxes,
                      ha='left', va='center',
                      bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.5),
                      fontsize=15, wrap=True)
    ax_map.grid(True)
    # Adjust the plot area
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.85, hspace=0.1)

    # Move legend outside the plot
    handles, labels = ax_map.get_legend_handles_labels()
    ax_map.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    frame_data = extract_frame_data(frame, day, vehicles, refuel_vehicles, hubs)
    update_csv(frame_data, day)
    return ax_map
def draw_refuel_dashboard(ax, vehicles):
    ax.clear()
    ax.axis('off')
    # Create a box for the fuel levels, positioned to the left
    box = ax.text(0.05, 0.5, '', transform=ax.transAxes,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', alpha=0.7),
                  ha='left', va='center')
    fuel_text = "Fuel Levels\n\n"
    for vehicle in vehicles:
        fuel_percentage = vehicle["fuel_level"] / vehicle["fuel_capacity"]
        bar = '█' * int(fuel_percentage * 10) + '▒' * (10 - int(fuel_percentage * 10))
        fuel_text += f"V{vehicle['id']}: {bar} {vehicle['fuel_level']:.0f}\n"
    box.set_text(fuel_text)
    box.get_bbox_patch().set_width(0.3)
    box.get_bbox_patch().set_height(0.2)
    return ax
def draw_patrol_areas(ax):
    for vehicle in vehicles:
        patrol_area = vehicle["patrol_area"]
        rect = plt.Rectangle((patrol_area[0], patrol_area[1]),
                             patrol_area[2] - patrol_area[0],
                             patrol_area[3] - patrol_area[1],
                             fill=False, edgecolor='red', linestyle='--', linewidth=2)
        ax.add_patch(rect)
def plot_entities(ax, vehicles, refuel_vehicles, hubs):
    # Draw patrol areas first so they appear behind other elements
    draw_patrol_areas(ax)
    for vehicle in vehicles:
        path = np.array(vehicle["path"])
        if len(path.shape) == 1:
            path = path.reshape(-1, 2)
        ax.plot(path[:, 0], path[:, 1], alpha=0.5)

        if vehicle["fuel_level"] <= 0:
            color = 'black'
            label = f'Vehicle {vehicle["id"]} (Stranded)'
        elif vehicle["fuel_level"] <= 30:
            color = 'red'
            label = f'Vehicle {vehicle["id"]} (Low Fuel)'
        elif vehicle["fuel_level"] <= 50:
            color = 'yellow'
            label = f'Vehicle {vehicle["id"]} (Medium Fuel)'
        else:
            color = 'green'
            label = f'Vehicle {vehicle["id"]} (High Fuel)'

        ax.scatter(vehicle["position"][0], vehicle["position"][1], color=color, s=100, marker='X', label=label)
        ax.text(vehicle["position"][0], vehicle["position"][1] + 0.5, f'{vehicle["fuel_level"]:.1f}', fontsize=8,
                ha='center')

    for refuel_vehicle in refuel_vehicles:
        ax.scatter(refuel_vehicle["position"][0], refuel_vehicle["position"][1], color='blue', s=100, marker='o',
                   label=f'Refuel Vehicle {refuel_vehicle["id"]}')
        if refuel_vehicle["refuel_point"] is not None and refuel_vehicle["state"] in ["moving_to_gather", "refueling"]:
            ax.scatter(refuel_vehicle["refuel_point"][0], refuel_vehicle["refuel_point"][1], color='cyan', s=50,
                       marker='s', label='Refuel Point')
            ax.plot([refuel_vehicle["position"][0], refuel_vehicle["refuel_point"][0]],
                    [refuel_vehicle["position"][1], refuel_vehicle["refuel_point"][1]], 'b--', alpha=0.5)

    for hub in hubs:
        ax.scatter(hub["position"][0], hub["position"][1], color='purple', s=200, marker='s', label=f'Hub {hub["id"]}')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
def run_simulation_day(day, progress_callback=None):
    global vehicles, refuel_vehicles, hubs

    frames = []
    all_data = []
    for frame in range(600):
        update(frame, day)
        frames.append(mplfig_to_npimage(fig))
        frame_data = extract_frame_data(frame, day, vehicles, refuel_vehicles, hubs)
        all_data.extend(frame_data)
        if progress_callback and frame % 60 == 0:  # Update every 60 frames
            progress_callback(f"Processing frame {frame}/600")

    plt.close(fig)
    daily_cost = calculate_daily_refueling_cost(vehicles, refuel_vehicles, hubs)
    return frames, daily_cost, all_data
def run_full_simulation(progress_callback=None, stop_event=None):
    global current_day, best_strategy, hubs
    daily_costs = []
    baseline = baseline_strategy()
    baseline_costs = []

    for day in range(1, DAYS + 1):
        if stop_event and stop_event.is_set():
            break

        current_day = day
        if progress_callback:
            progress_callback(f"Day {day}: Updating hub positions")

        # Update hub positions for this day
        day_hub_positions = all_hub_positions[day - 1]
        for i, hub in enumerate(hubs):
            hub["position"] = day_hub_positions[i]

        # Reset vehicles for the new day
        for vehicle in vehicles:
            vehicle["fuel_level"] = vehicle["fuel_capacity"]
            vehicle["state"] = "patrolling"
            patrol_area = vehicle["patrol_area"]
            vehicle["position"] = np.array([(patrol_area[0] + patrol_area[2]) / 2,
                                            (patrol_area[1] + patrol_area[3]) / 2])
            vehicle["path"] = [vehicle["position"].tolist()]

        # Reset refuel vehicles for the new day
        for i, rv in enumerate(refuel_vehicles):
            rv["position"] = hubs[i]["position"].copy()
            rv["hub"] = hubs[i]
            rv["state"] = "idle"
            rv["path"] = []
            rv["refuel_point"] = None

        _, cost = run_simulation_day(progress_callback)
        daily_costs.append(cost)
        baseline_cost = calculate_daily_refueling_cost(baseline)
        baseline_costs.append(baseline_cost)
    return daily_costs, baseline_costs
def reset_vehicles_and_refuelers():
    for vehicle in vehicles:
        vehicle["fuel_level"] = vehicle["fuel_capacity"]
        vehicle["state"] = "patrolling"
        patrol_area = vehicle["patrol_area"]
        vehicle["position"] = np.array([(patrol_area[0] + patrol_area[2]) / 2,
                                        (patrol_area[1] + patrol_area[3]) / 2])
        vehicle["path"] = [vehicle["position"].tolist()]

    for i, rv in enumerate(refuel_vehicles):
        rv["position"] = hubs[i]["position"].copy()
        rv["hub"] = hubs[i]
        rv["state"] = "idle"
        rv["path"] = []
        rv["refuel_point"] = None
def baseline_strategy():
    return {
        "dispatch_strategy": [random.choice(hubs)["id"] for _ in vehicles],  # Random hub assignment
        "gather_strategy": [False for _ in vehicles],  # No gathering
        "refuel_threshold": 0.2,  # Lower threshold
        "emergency_threshold": 0.1  # Lower emergency threshold
    }

def apply_baseline_strategy(strategy):
    for vehicle in vehicles:
        if vehicle["fuel_level"] <= strategy["refuel_threshold"] * vehicle["fuel_capacity"]:
            if random.random() < 0.3:  # 30% chance to choose nearest hub
                nearest_hub = min(hubs, key=lambda h: np.linalg.norm(h["position"] - vehicle["position"]))
                vehicle["path"] = shortest_path(vehicle["position"], nearest_hub["position"])
            else:
                random_hub = random.choice(hubs)
                vehicle["path"] = shortest_path(vehicle["position"], random_hub["position"])
            vehicle["state"] = "moving_to_hub"
        elif vehicle["fuel_level"] <= strategy["emergency_threshold"] * vehicle["fuel_capacity"]:
            vehicle["state"] = "emergency_refuel"
            vehicle["fuel_level"] = vehicle["fuel_capacity"] * random.uniform(0.4, 0.6)

def calculate_baseline_cost(vehicles, hubs):
    total_cost = 0.0  # Initialize with 0.0 to ensure it's a float
    for vehicle in vehicles:
        # Randomize movement cost
        movement_cost_factor = random.uniform(12, 18)
        total_cost += len(vehicle["path"]) * vehicle["consumption_rate"] * movement_cost_factor

        # Randomize refueling cost
        if vehicle["state"] in ["moving_to_hub", "emergency_refuel"]:
            refuel_amount = vehicle["fuel_capacity"] - vehicle["fuel_level"]
            refuel_cost_factor = random.uniform(1.3, 1.7)
            total_cost += refuel_amount * refuel_cost_factor

        # Randomize operational cost
        operational_cost = random.uniform(1300, 1700)
        total_cost += operational_cost

    # Randomize hub maintenance cost
    hub_maintenance_cost = random.uniform(7000, 8000)
    total_cost += len(hubs) * hub_maintenance_cost

    # Add random unexpected costs
    if random.random() < 0.2:  # 20% chance of unexpected cost
        unexpected_cost = random.uniform(1000, 5000)
        total_cost += unexpected_cost

    return max(total_cost, 1.0)
#main
current_day = 0
daily_costs = []
baseline_daily_costs = []
videos = []

fig, ax = plt.subplots(figsize=(16, 12))

ensure_videos_directory()
for day in range(1, DAYS + 1):
    print(f"Simulating Day {day}")

    current_day = day

    # Update hub positions for this day
    day_hub_positions = all_hub_positions[day - 1]
    for i, hub in enumerate(hubs):
        hub["position"] = day_hub_positions[i]

    # Reset vehicles and refuel vehicles for the new day
    reset_vehicles_and_refuelers()

    # Run the genetic algorithm to find the best strategy for this day's hub configuration
    best_strategy, _ = genetic_algorithm()

    # Apply the best strategy and run the simulation
    apply_refueling_strategy(best_strategy)
    frames, daily_cost, day_data = run_simulation_day(day)
    update_csv(day_data, day)
    # Optimized strategy
    reset_vehicles_and_refuelers()
    best_strategy, _ = genetic_algorithm()
    apply_refueling_strategy(best_strategy)
    frames, daily_cost, day_data = run_simulation_day(day)
    optimized_cost = calculate_daily_refueling_cost(vehicles, refuel_vehicles, hubs)
    daily_costs.append(optimized_cost)

    # Baseline strategy
    reset_vehicles_and_refuelers()
    baseline = baseline_strategy()
    apply_baseline_strategy(baseline)
    _, _, _ = run_simulation_day(day)  # Run simulation with baseline strategy
    baseline_cost = calculate_baseline_cost(vehicles, hubs)
    if baseline_cost is None:
        print(f"Warning: Baseline cost for day {day} is None. Using default value.")
        baseline_cost = optimized_cost * 1.2  # Use a default value if None
    baseline_daily_costs.append(baseline_cost)

    if isinstance(optimized_cost, (int, float)) and isinstance(baseline_cost, (int, float)):
        cost_difference = baseline_cost - optimized_cost
        cost_saving_percentage = (cost_difference / baseline_cost) * 100 if baseline_cost > 0 else 0

        print(f"Day {day} Optimized Cost: ₹{optimized_cost:,.2f}")
        print(f"Day {day} Baseline Cost: ₹{baseline_cost:,.2f}")
        print(f"Cost Saving: ₹{cost_difference:,.2f} ({cost_saving_percentage:.2f}%)")
    else:
        print(f"Warning: Invalid cost values for day {day}. Optimized: {optimized_cost}, Baseline: {baseline_cost}")


    # Create and save video (using the optimized strategy results)
    def make_frame(t):
        return frames[int(t * 2)]

    clip = VideoClip(make_frame, duration=len(frames) / 2)
    filename = os.path.join("Videos", f"day_{day}_simulation.mp4")
    clip.write_videofile(filename, fps=2)
    videos.append(filename)
    print(f"Video for day {day} saved as {filename}")

print("All daily simulation videos have been created.")

# Create the cost comparison graph
plt.figure(figsize=(12, 6))
plt.plot(range(1, DAYS + 1), daily_costs, marker='o', label='Optimized Strategy')
plt.plot(range(1, DAYS + 1), baseline_daily_costs, marker='s', label='Baseline Strategy')
plt.title('Daily Refueling Costs: Optimized vs Baseline')
plt.xlabel('Day')
plt.ylabel('Cost (₹)')
plt.legend()
plt.grid(True)

# Ensure Analysis directory exists
os.makedirs("Analysis", exist_ok=True)

# Create a unique filename using current date and time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
graph_filename = f"cost_comparison_{current_time}.png"

# Save the graph with the unique filename
plt.savefig(os.path.join("Analysis", graph_filename))
plt.close()

print(f"Cost comparison graph saved as: {graph_filename}")

# Calculate and save cost improvement metrics
total_optimized_cost = sum(daily_costs)
total_baseline_cost = sum(baseline_daily_costs)
total_cost_saving = total_baseline_cost - total_optimized_cost
cost_saving_percentage = (total_cost_saving / total_baseline_cost) * 100

with open("Analysis/cost_metrics.txt", "w") as f:
    f.write(f"Total Optimized Cost: {total_optimized_cost:,.2f}\n")
    f.write(f"Total Baseline Cost: {total_baseline_cost:,.2f}\n")
    f.write(f"Total Cost Saving: {total_cost_saving:,.2f}\n")
    f.write(f"Cost Saving Percentage: {cost_saving_percentage:.2f}%\n")

print("Analysis data has been saved in the Analysis folder.")
if __name__ == "__main__":
    print("Main simulation completed. Starting analysis...")

    # Import and run the analysis
    import analysis

    analysis.analyze_simulation()

    print("Analysis completed.")
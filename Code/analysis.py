import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
from scipy import stats
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Function to generate unique filenames
def get_unique_filename(base_path, base_name, extension):
    counter = 1
    while True:
        new_filename = f"{base_name}_{counter}{extension}"
        full_path = os.path.join(base_path, new_filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1


# Read and concatenate all CSV files
csv_files = glob.glob('Simulation_Data/simulation_data_day_*.csv')
dfs = [pd.read_csv(file) for file in csv_files]
full_df = pd.concat(dfs, ignore_index=True)

# Convert frame to time (assuming 30 frames per minute)
full_df['time'] = full_df['frame'] / 30 / 60  # time in hours


def analyze_simulation():
    insights = []

    # 1. Fuel Efficiency Analysis
    vehicle_data = full_df[full_df['entity_type'] == 'vehicle']
    fuel_efficiency = vehicle_data.groupby('id', group_keys=False).apply(
        lambda x: x['fuel_level'].diff().div(x['time'].diff()).mean(), include_groups=False
    )

    if isinstance(fuel_efficiency, pd.Series) and not fuel_efficiency.empty:
        most_efficient = fuel_efficiency.idxmax()
        least_efficient = fuel_efficiency.idxmin()
        insights.append(f"Most fuel-efficient vehicle: {most_efficient}")
        insights.append(f"Least fuel-efficient vehicle: {least_efficient}")
    else:
        insights.append("Unable to determine fuel efficiency due to insufficient data")

    # 2. Busiest Refueling Spots
    refuel_spots = full_df[full_df['state'] == 'refueling'].groupby(
        ['refuel_point_x', 'refuel_point_y']).size().sort_values(ascending=False)
    if not refuel_spots.empty:
        busiest_spot = refuel_spots.index[0]
        insights.append(f"Busiest refueling spot: {busiest_spot} with {refuel_spots.iloc[0]} refuels")
    else:
        insights.append("No refueling activity detected")

    # 3. Vehicle State Transitions
    state_transitions = vehicle_data.groupby('id')['state'].apply(lambda x: x.value_counts()).sum()
    if isinstance(state_transitions, pd.Series) and not state_transitions.empty:
        most_common_transition = state_transitions.idxmax()
        insights.append(f"Most common vehicle state: {most_common_transition}")
    else:
        insights.append("No state transitions detected or insufficient data")

    # 4. Stranded Vehicle Analysis
    stranded_events = vehicle_data[vehicle_data['state'] == 'stranded']
    if not stranded_events.empty:
        stranded_locations = stranded_events.groupby(['x_position', 'y_position']).size().sort_values(ascending=False)
        if not stranded_locations.empty:
            danger_zone = stranded_locations.index[0]
            insights.append(f"Most dangerous zone (frequent stranding): {danger_zone}")
        else:
            insights.append("Stranded vehicles detected, but unable to determine dangerous zones")
    else:
        insights.append("No stranded vehicles detected")

    # 5. Refuel Vehicle Efficiency
    refuel_vehicle_data = full_df[full_df['entity_type'] == 'refuel_vehicle']
    refuel_efficiency = refuel_vehicle_data.groupby('id')['state'].apply(lambda x: (x == 'refueling').sum())
    if isinstance(refuel_efficiency, pd.Series) and not refuel_efficiency.empty:
        most_efficient_refueler = refuel_efficiency.idxmax()
        insights.append(
            f"Most efficient refuel vehicle: {most_efficient_refueler} with {refuel_efficiency.max()} refuels")
    else:
        insights.append("No refuel vehicle activity detected or insufficient data")

    # 6. Path Complexity Analysis
    path_complexity = vehicle_data.groupby('id', group_keys=False).apply(
        lambda x: np.mean(np.diff(x['x_position']) ** 2 + np.diff(x['y_position']) ** 2), include_groups=False
    )
    if isinstance(path_complexity, pd.Series) and not path_complexity.empty:
        most_complex_path = path_complexity.idxmax()
        insights.append(f"Vehicle with most complex path: {most_complex_path}")
    else:
        insights.append("Unable to determine path complexity")

    # 7. Fuel Level Distribution
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=vehicle_data, x='fuel_level', hue='day', palette='viridis')
    plt.title('Fuel Level Distribution Across Days')
    fuel_dist_file = get_unique_filename('Simulation_Data/Insights', 'fuel_level_distribution', '.png')
    plt.savefig(fuel_dist_file)
    plt.close()

    # 8. Vehicle Position Heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.kdeplot(data=vehicle_data, x='x_position', y='y_position', cmap='YlOrRd', fill=True, cbar=False)
    plt.title('Heatmap of Vehicle Position Density')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    mappable = heatmap.collections[0]
    cbar = plt.colorbar(mappable)
    cbar.set_label('Relative Density of Vehicle Positions')
    heatmap_file = get_unique_filename('Simulation_Data/Insights', 'vehicle_position_heatmap', '.png')
    plt.savefig(heatmap_file)
    plt.close()

    # 9. Correlation between Fuel Level and Distance from Hub
    hubs = full_df[full_df['entity_type'] == 'hub'][['x_position', 'y_position']].drop_duplicates()
    if not hubs.empty:
        vehicle_data = vehicle_data.copy()
        vehicle_data['distance_from_hub'] = vehicle_data.apply(lambda row:
                                                               np.min([np.sqrt(
                                                                   (row['x_position'] - hub['x_position']) ** 2 + (
                                                                           row['y_position'] - hub[
                                                                       'y_position']) ** 2) for _, hub in
                                                                   hubs.iterrows()]),
                                                               axis=1)
        correlation = vehicle_data['fuel_level'].corr(vehicle_data['distance_from_hub'])
        insights.append(f"Correlation between fuel level and distance from hub: {correlation:.2f}")
    else:
        insights.append("No hub data available for correlation analysis")

    # 10. K-means Clustering Analysis
    # Prepare data for clustering
    cluster_data = vehicle_data[['x_position', 'y_position', 'fuel_level']].dropna()

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Perform K-means clustering
    n_clusters = 5  # You can adjust this number
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Add cluster labels to the data
    cluster_data['Cluster'] = cluster_labels

    # Visualize the clusters
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(cluster_data['x_position'], cluster_data['y_position'],
                          c=cluster_data['Cluster'], cmap='viridis', alpha=0.6)
    plt.title('K-means Clustering of Vehicles')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # Annotate patrol areas
    plt.axhline(y=20, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=20, color='r', linestyle='--', linewidth=1)
    plt.text(10, 30, 'Patrol 1', fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.4))
    plt.text(30, 30, 'Patrol 2', fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.4))
    plt.text(10, 10, 'Patrol 3', fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.4))
    plt.text(30, 10, 'Patrol 4', fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.4))

    # Save the figure
    cluster_file = get_unique_filename('Simulation_Data/Insights', 'vehicle_clusters', '.png')
    plt.savefig(cluster_file)
    plt.close()

    # Analyze clusters
    cluster_summary = cluster_data.groupby('Cluster').agg({
        'x_position': 'mean',
        'y_position': 'mean',
        'fuel_level': 'mean'
    }).round(2)

    insights.append(f"K-means clustering identified {n_clusters} vehicle clusters:")
    for cluster, summary in cluster_summary.iterrows():
        insights.append(f"Cluster {cluster}: Center ({summary['x_position']}, {summary['y_position']}), "
                        f"Avg Fuel Level: {summary['fuel_level']}")

    return insights, fuel_dist_file, heatmap_file, cluster_file


# Run the analysis
insights, fuel_dist_file, heatmap_file, cluster_file = analyze_simulation()

# Save insights to a text file
insights_file = get_unique_filename('Simulation_Data/Insights', 'simulation_insights', '.txt')
with open(insights_file, 'w') as f:
    for insight in insights:
        f.write(f"- {insight}\n")

print(f"Analysis completed. Insights saved in {insights_file}")
print(f"Fuel level distribution plot saved as {fuel_dist_file}")
print(f"Vehicle position heatmap saved as {heatmap_file}")
print(f"K-means clustering plot saved as {cluster_file}")
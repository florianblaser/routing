import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import random

from scipy.spatial import distance
import plotly.graph_objs as go

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import plotly.graph_objs as go

def plot_all_nodes_with_angles(nodes_df, home_node_id=0):
    """Plot all nodes with their respective details displayed on hover, with the home node colored distinctly."""
    fig = go.Figure()

    # Assign colors and sizes based on node ID, with a special color for the home node
    colors = ['#00BFC4' if node_id != home_node_id else '#FF6347' for node_id in nodes_df.index]
    sizes = [10 if node_id != home_node_id else 12 for node_id in nodes_df.index]  # Home node slightly larger

    # Add all nodes to the plot
    fig.add_trace(go.Scatter(
        x=nodes_df['x'],
        y=nodes_df['y'],
        mode='markers',
        marker=dict(color=colors, size=sizes),
        text=[
            f"Node: {node_id}<br>X: {row['x']}<br>Y: {row['y']}<br>Angle: {row['angle_to_home']}"
            for node_id, row in nodes_df.iterrows()
        ],
        hoverinfo='text'
    ))

    # Update the plot layout to make it more rectangular
    fig.update_layout(
        title="Plot of All Nodes with Angles",
        xaxis=dict(title="X Coordinate", range=[nodes_df['x'].min() - 5, nodes_df['x'].max() + 5]),
        yaxis=dict(title="Y Coordinate", range=[nodes_df['y'].min() - 5, nodes_df['y'].max() + 5], scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        hovermode='closest',
        height=500,
        width=700
    )

    # Show the figure
    fig.show()
    
def create_nodes_dataframe(num_nodes, home_node_id, min_work_days, days_off, visiting_interval_min=10, visiting_interval_max=30, max_last_visit=30, frac_fixed_app=.1, simple_schedule=False):
    """Create a DataFrame containing nodes' information, including their attributes and distances."""
    node_ids = np.arange(0, num_nodes)
    
    def generate_opening_hours(min_work_days):
        """Generate opening hours for a random subset of days."""
        schedule = {}
        # Randomly select a minimal number of working days within a given week
        days = sorted(random.sample(range(1, 8), min_work_days))
        for day in days:
            # Generate random start hours and end hours within constraints
            start_hour = 8 + random.randint(0, 1)
            end_hour = start_hour + random.randint(8, 9)
            start_minute = random.randint(0, 59)
            end_minute = random.randint(0, 59)

            # Randomly add lunch break
            if random.random() < 0.5:
                lunch_start = 11 + random.randint(0, 1)
                lunch_end = lunch_start + 1
                lunch_start_minute = random.randint(0, 59)
                lunch_end_minute = random.randint(0, 59)

                # Creating datetime.time objects for each period
                schedule[day] = [
                    [time(start_hour, start_minute), time(lunch_start, lunch_start_minute)],
                    [time(lunch_end, lunch_end_minute), time(end_hour, end_minute)]
                ]
            else:
                # Creating datetime.time objects for a non-interrupted period
                schedule[day] = [
                    [time(start_hour, start_minute), time(end_hour, end_minute)]
                ]
            if simple_schedule:
                schedule[day] = [[time(8, 0), time(18, 0)]]
        return schedule
    
    def convert_to_time(hours_dict): # duplicate
        """Convert string-based opening hours to datetime.time objects."""
        return {k: (datetime.strptime(v[0], "%H:%M").time(), datetime.strptime(v[1], "%H:%M").time()) for k, v in hours_dict.items()}

    def add_fixed_appointments(nodes_df, frac_fixed_app):
        selected_indices = nodes_df.sample(frac=frac_fixed_app, replace=True).index
        for idx in selected_indices:
            # day can't be in days_off
            day = random.choice([day for day in nodes_df.at[idx, 'opening_hours'].keys() if day not in days_off])
            start, end = nodes_df.at[idx, 'opening_hours'][day][0][0], nodes_df.at[idx, 'opening_hours'][day][-1][-1]
            if isinstance(start, str):  # Checking if start time is still a string
                start = datetime.strptime(start, "%H:%M").time()
            if isinstance(end, str):  # Checking if end time is still a string
                end = datetime.strptime(end, "%H:%M").time()
            start_dt = datetime.combine(datetime.today(), start)
            end_dt = datetime.combine(datetime.today(), end)
            appointment_start = start_dt + timedelta(minutes=random.randint(0, (end_dt - start_dt).seconds // 60 - 30))
            appointment_end = appointment_start + timedelta(minutes=30)
            nodes_df.at[idx, 'fixed_appointment'].append([day, appointment_start.time(), appointment_end.time()])
        return nodes_df
    
    current_date = datetime.now()
    last_visited = [current_date - timedelta(days=np.random.randint(1, max_last_visit)) for _ in range(num_nodes)]
    last_visited = [date.strftime("%Y-%m-%d") for date in last_visited]
    visiting_intervals = np.random.randint(visiting_interval_min, visiting_interval_max, size=num_nodes)
    durations = np.random.randint(30, 61, size=num_nodes)

    # Consolidate various attributes into a DataFrame for each node
    nodes_df = pd.DataFrame({
        "node_id": node_ids,
        "opening_hours": [generate_opening_hours(min_work_days) for _ in range(num_nodes)],
        "last_visited": last_visited,
        "Visiting Interval (days)": visiting_intervals,
        "on_site_time": durations
    })

    nodes_df.loc[nodes_df['node_id'] == 0, 'on_site_time'] = 0

    nodes_df.sort_values("node_id", inplace=True)
    nodes_df['days_since_last_visit'] = (current_date - pd.to_datetime(nodes_df['last_visited'])).dt.days
    nodes_df['priority'] = nodes_df['days_since_last_visit'] / nodes_df['Visiting Interval (days)']
    nodes_df['priority'] = nodes_df['priority'].apply(lambda x: 1 if x > 1 else x)
    nodes_df['fixed_appointment'] =  [[] for _ in range(num_nodes)]
    nodes_df['opening_hours'] = nodes_df['opening_hours']
    nodes_df = add_fixed_appointments(nodes_df, frac_fixed_app)

    # Generate random coordinates for each node
    coordinates = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_nodes)]
    coordinates_df = pd.DataFrame(coordinates, columns=['x', 'y'])
    nodes_df = pd.concat([nodes_df, coordinates_df], axis=1)

    def calculate_angles(nodes_df, home_x, home_y):
        """
        Calculate angles between the home node and other nodes.
        This helps in sorting nodes into angular clusters.
        """
        return np.degrees(np.arctan2(nodes_df['y'] - home_y, nodes_df['x'] - home_x)) % 360

    home_node = nodes_df.loc[home_node_id]
    home_x, home_y = home_node['x'], home_node['y']

    # Calculate the angles of all nodes with respect to the home node
    angles = calculate_angles(nodes_df, home_x, home_y)
    nodes_df['angle_to_home'] = angles

    nodes_df = nodes_df.explode('fixed_appointment').reset_index(drop=True)
    nodes_df['weekday_fixed_appointment'] = nodes_df['fixed_appointment'].apply(lambda x: x[0] if isinstance(x, list) else None)
    
    # ensure
    nodes_df['original_node_id'] = nodes_df['node_id']
    nodes_df.reset_index(drop=True, inplace=True)
    nodes_df['node_id'] = nodes_df.index

    # Create a distance matrix based on Euclidean distance
    def calculate_euclidean_distance_matrix(coords):
        return distance.cdist(coords, coords, 'euclidean')

    distance_matrix = calculate_euclidean_distance_matrix(nodes_df[['x', 'y']].values)
    distance_df = pd.DataFrame(np.round(distance_matrix, 2), columns=range(0, len(nodes_df)), index=range(0, len(nodes_df)))

    # Ensure the diagonal is zero to represent self-distance
    np.fill_diagonal(distance_df.values, 0)

    # add "dist_to_home" column
    nodes_df['dist_to_home'] = distance_df[home_node_id]

    all_days = set(range(1, 8))
    # Calculate open days as sets from the dictionary keys
    nodes_df['open_days'] = nodes_df['opening_hours'].apply(lambda x: set(x.keys()))

    # Calculate closed days
    nodes_df['closed_days'] = nodes_df['open_days'].apply(lambda x: set(all_days - x))

    # nodes with a fixed appointment have priorty = 1
    nodes_df['priority'] = nodes_df.apply(lambda row: 9 if isinstance(row['fixed_appointment'], list) else row['priority'], axis=1)
    return nodes_df, distance_df
 
def calculate_metric(distance_matrix, nodes_df, global_max_dist, node_ids, print_ind_metrics=True):
    filtered_nodes_df = nodes_df[nodes_df['node_id'].isin(node_ids)]
    
    # Identify indices corresponding to the filtered node IDs
    # index_map = {node_id: index for index, node_id in enumerate(nodes_df['node_id'])}
    # filtered_indices = [index_map[node_id] for node_id in node_ids]
    # Extract relevant submatrix from the distance matrix
    # filtered_distance_matrix = distance_matrix[np.ix_(filtered_indices, filtered_indices)]
    # Calculate mean and variance of distances between nodes using the distance matrix
    # distances = filtered_distance_matrix[np.triu_indices_from(filtered_distance_matrix, k=1)]
    # mean_dist_between_nodes = np.mean(distances)
    # var_dist_between_nodes = np.var(distances, ddof=1)

    num_nodes_metric = len(filtered_nodes_df) / len(nodes_df)
    
    priority_metric = filtered_nodes_df['priority'].nlargest(int(0.3 * len(filtered_nodes_df))).mean()
    
    max_dist_to_root = filtered_nodes_df['dist_to_home'].max()
    dist_metric = max_dist_to_root / global_max_dist

    # prevent any metric from being nan
    # if np.isnan(num_nodes_metric):
    #     num_nodes_metric = 0.5
    if np.isnan(priority_metric):
        priority_metric = 0.5
    # if np.isnan(dist_metric):
    #     dist_metric = 0.5

    metric = (num_nodes_metric + dist_metric) / 2

    if print_ind_metrics:
        print(f"Number of nodes metric: {num_nodes_metric}")
        print(f"Priority metric: {priority_metric}")
        print(f"Distance metric: {dist_metric}")
        print(f"Overall metric: {metric}")
    
    return metric

def adjust_angles(clusters, nodes_df, angle_sizes, degree_adj, global_max_dist, num_small_clusters, num_large_clusters, overnight_factor, distance_matrix, verbose):    
    metrics = {}
    for cluster_id, node_ids in clusters.items():
        metric = calculate_metric(distance_matrix, nodes_df, global_max_dist, node_ids, print_ind_metrics=False)
        metrics[cluster_id] = metric

    metric_sum = sum(metrics.values())
    small_soll = metric_sum / (num_large_clusters * overnight_factor + num_small_clusters)
    large_soll = small_soll * overnight_factor

    deviations = {cluster_id: metrics[cluster_id] - (small_soll if 'small' in cluster_id else large_soll) for cluster_id in clusters}
    
    # assigne the degree_adj to the clusters based on the deviation
    for cluster_id, deviation in deviations.items():
        angle_sizes[cluster_id] -= deviation * degree_adj

    if verbose:
        print("Deviations, metrics and new angle sizes:")
        for cluster_id in clusters:
            print(f"Cluster {cluster_id} \
                with deviation {round(deviations[cluster_id], 2)} \
                and metric {round(metrics[cluster_id], 2)} \
                has new angle size {round(angle_sizes[cluster_id], 2)} \
                spanning from {round(angle_sizes[cluster_id] - deviations[cluster_id] * degree_adj, 2)}° to {round(angle_sizes[cluster_id], 2)}°.")
        
    return angle_sizes

def custom_clustering(distance_matrix, nodes_df, num_small_clusters, num_large_clusters, overnight_factor, precision, home_node_id=0, verbose=False):
    # remove the home node from the nodes_df
    if nodes_df.index[0] == 0:
        nodes_df_copy = nodes_df.drop(0).copy()
    
    # define the largest gap to limit the span of all clusters
    clusters = {f'small_{i}': [home_node_id] for i in range(num_small_clusters)}
    clusters.update({f'large_{i}': [home_node_id] for i in range(num_large_clusters)})
    
    angles = sorted(nodes_df_copy['angle_to_home'])
    diffs = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    diffs.append(360 - angles[-1] + angles[0])
    
    max_gap = max(diffs)
    gap_start = angles[diffs.index(max_gap)]
    gap_end = angles[(diffs.index(max_gap) + 1) % len(angles)]

    if verbose == True:
        print(f"Largest gap spans from {gap_start}° to {gap_end}°, covering {max_gap}°.")

    cluster_start = gap_end
    num_clusters = num_small_clusters + num_large_clusters

    # Initialize the angle sizes for the clusters
    total_span = (360 - max_gap)
    degree_adj = total_span / 10
    small_step = total_span / (num_small_clusters + num_large_clusters * overnight_factor)
    large_step = small_step * overnight_factor

    if verbose == True:
        print(f"Small step size is {small_step}°, large step size is {large_step}°.")

    angle_sizes = {}
    for i in range(num_clusters):
        if i < num_small_clusters:
            angle_sizes[f'small_{i}'] = small_step
        else:
            angle_sizes[f'large_{i-num_small_clusters}'] = large_step

    for key, size in angle_sizes.items():
        if verbose == True:
            print(f"Cluster {key} spans {size}°")
    
    global_max_dist = nodes_df_copy['dist_to_home'].max()

    # Initial assignment of nodes to clusters
    for i in range(precision):
        current_angle = cluster_start  # Reset the start angle for each precision iteration
            
        # add the home node to each cluster
        for key in clusters.keys():
            clusters[key] = [home_node_id]

        # Assign nodes to clusters based on their angle to the home node
        for cluster_id, size in angle_sizes.items():
            start_angle = current_angle
            # round up to the nearest integer
            start_angle = int(start_angle)
            end_angle = (current_angle + size) % 360
            end_angle = int(np.ceil(end_angle))
            # Ensuring all nodes are assigned, handling the wrap-around scenario more cleanly
            if end_angle < start_angle:  # This handles the case where the segment wraps past 360 degrees
                nodes_in_cluster = [index for index, row in nodes_df_copy.iterrows() if 
                                    (row['angle_to_home'] >= start_angle or row['angle_to_home'] < end_angle)]
            else:  # No wrap-around, normal case
                nodes_in_cluster = [index for index, row in nodes_df_copy.iterrows() if 
                                    (start_angle <= row['angle_to_home'] < end_angle)]

            clusters[cluster_id] = [home_node_id] + nodes_in_cluster
            current_angle = end_angle

        if (i == 0) & (verbose == True):
            print("Initial clusters:")
            for key, value in clusters.items():
                print(f"Cluster {key} with nodes {value}")

        angle_sizes = adjust_angles(clusters, nodes_df_copy, angle_sizes, degree_adj, global_max_dist, num_small_clusters, num_large_clusters, overnight_factor, distance_matrix, verbose)
        degree_adj *= 0.9

        #plot_refined_clusters(clusters, nodes_df)
        
    return clusters

def assign_weekdays_to_clusters(nodes_df):
    nodes_df['weekdays_fixed_appointments'] = nodes_df['fixed_appointment'].apply(lambda x: x[0] if isinstance(x, tuple) else None)
    grouped_data = nodes_df.groupby(['weekdays_fixed_appointments', 'cluster']).size().unstack(fill_value=0)
    weekdays = [0.0, 1.0, 2.0, 3.0, 4.0]

    cluster_assignments = {}

    # Identify missing weekdays
    missing_weekdays = set(weekdays) - set(grouped_data.index)

    # Add missing weekdays as new rows filled with zeros
    for weekday in missing_weekdays:
        grouped_data.loc[weekday] = [0] * len(grouped_data.columns)  # Create a row of zeros

    # Sort the DataFrame by index to ensure weekdays are in order
    grouped_data.sort_index(inplace=True)

    # Generate priority list focusing on appropriate assignment of large clusters
    priority_list = []
    for cluster in grouped_data.columns:
        for i in range(len(weekdays)):
            if 'large' in cluster and i < len(weekdays) - 1:  # ensure there's a next day for large clusters
                avg_appointments = (grouped_data.loc[weekdays[i], cluster] + grouped_data.loc[weekdays[i+1], cluster]) / 2
                priority_list.append((weekdays[i], cluster, avg_appointments, f"Days {weekdays[i]}-{weekdays[i+1]}"))
            elif 'small' in cluster:
                priority_list.append((weekdays[i], cluster, grouped_data.loc[weekdays[i], cluster], f"Day {weekdays[i]}"))

    # Sort priority list
    priority_df = pd.DataFrame(priority_list, columns=['day', 'cluster', 'appointments', 'description'])
    priority_df = priority_df.sort_values(by=['cluster', 'appointments'], ascending=[True, False])
    all_clusters = nodes_df['cluster'].unique()
    large_clusters = [cluster for cluster in all_clusters if 'large' in cluster]
    large_clusters_not_in_priority = [cluster for cluster in large_clusters if cluster not in set(priority_df['cluster'])]

    # check if there is any entry in priority_df['description'] containing 'Days'
    for index, large_cluster in enumerate(large_clusters_not_in_priority):
        # create a list of consecutive days based on weekdays
        consecutive_day_pairs = []
        for i in range(len(weekdays)):
            if i < len(weekdays) - 1:
                consecutive_day_pairs.append((weekdays[i], weekdays[i+1]))

        # get the number of appointments for each pair of consecutive_days
        appointments_per_pair = {}
        try:
            for consecutive_days in consecutive_day_pairs:
                day1 = sum(priority_df[priority_df['day'] == consecutive_days[0]]['appointments'])
                day2 = sum(priority_df[priority_df['day'] == consecutive_days[1]]['appointments'])
                appointments_per_pair[consecutive_days] = day1 + day2

            # get the pair of consecutive days with the least number of appointments
            min_appointments = min(appointments_per_pair.values())

            # get the pair of consecutive days with the least number of appointments
            min_appointments_days = [k for k, v in appointments_per_pair.items() if v == min_appointments]

        except:
            min_appointments_days = [(0.0, 1.0), (2.0, 3.0)]

        cluster_assignments[large_cluster] = min_appointments_days[index]

    # Assign clusters to days
    used_days = set()
    for _, row in priority_df.iterrows():
        cluster = row['cluster']
        description = row['description']

        if 'Days' in description and cluster not in cluster_assignments:
            day1, day2 = map(float, description.split(' ')[1].split('-'))
            if day1 not in used_days and day2 not in used_days:
                cluster_assignments[cluster] = {day1, day2}
                used_days.update([day1, day2])
        elif 'Day' in description and cluster not in cluster_assignments and float(description.split(' ')[1]) not in used_days:
            day = float(description.split(' ')[1])
            cluster_assignments[cluster] = {day,}
            used_days.add(day)

    # find the clusters that are not assigned to any day
    unassigned_clusters = set(all_clusters) - set(cluster_assignments.keys())
    unassigned_days = set(weekdays) - used_days

    # randomly assign unassigned clusters to unassigned days
    for cluster in unassigned_clusters:
        day = unassigned_days.pop()
        cluster_assignments[cluster] = {day,}
    
    # Add a new column to nodes_df mapping each node's cluster to the weekday
    nodes_df['visit_day'] = nodes_df['cluster'].map(cluster_assignments)

    def update_visit_days(row):
        fixed_day = row['weekdays_fixed_appointments']
        visit_days = row['visit_day']
        
        # If there's a fixed appointment day and it's not in visit days, update the visit days
        if not pd.isna(fixed_day) and fixed_day not in visit_days:
            unique_visit_days = nodes_df['visit_day'].apply(lambda x: tuple(sorted(x))).unique()
            for unique_days in unique_visit_days:
                if fixed_day in unique_days:
                    return set(unique_days)
        return visit_days

    # Apply the function to the dataframe
    nodes_df['visit_day'] = nodes_df.apply(update_visit_days, axis=1)
    
    return nodes_df

def create_data_model(sub_distance_matrix):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = sub_distance_matrix
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data

def extract_solution_as_list(data, manager, routing, solution):
    """Extracts solution as a list of node indices."""
    route_list = []
    vehicle_id = 1
    index = routing.Start(vehicle_id)
    while not routing.IsEnd(index):
        route_list.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route_list.append(manager.IndexToNode(index))  # Add the last node
    return route_list

def find_route(clusters, nodes_df, distance_matrix, max_travel_distance=2000, span_cost_coefficient=100, slack=0):
    """Entry point of the program."""
    solutions = {}
    for cluster in clusters:
        nodes = clusters[cluster]
        sub_nodes_df = nodes_df[nodes_df['node_id'].isin(nodes)].reset_index(drop=True)
        sub_distance_matrix = distance_matrix.loc[nodes, nodes].values.tolist()
        sub_distance_matrix = [[int(x) for x in row] for row in sub_distance_matrix]
        node_mapping = sub_nodes_df['node_id'].to_dict()
        
        data = create_data_model(sub_distance_matrix)

        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            slack,  # no slack
            max_travel_distance,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(span_cost_coefficient)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            solution_list = extract_solution_as_list(data, manager, routing, solution)
            mapped_solution_list = [node_mapping[node] for node in solution_list]
            solutions[cluster] = mapped_solution_list
        else:
            solutions[cluster] = 'No solution found.'
    
    return solutions

def plot_ind_route(route_list, nodes_df, home_node_id=0):
    """Plot the solution path and clusters with distinct color schemes and highlight the home node distinctly."""    
    # Initialize Plotly figure
    fig = go.Figure()

    # Color palette for the route
    colors = ['#00BFC4']  # You can use a different color or a palette if you prefer

    # Plotting the solution route using mapped node IDs
    x = [nodes_df.loc[node_id, 'x'] for node_id in route_list]
    y = [nodes_df.loc[node_id, 'y'] for node_id in route_list]

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color=colors[0], width=2),
        marker=dict(color=colors[0], size=8),
        name='Route',
        hoverinfo='text',
        hovertext=[
            f"Node: {node_id}<br>X: {nodes_df.loc[node_id, 'x']}<br>Y: {nodes_df.loc[node_id, 'y']}"
            for node_id in route_list
        ]
    ))

    # Add the home node distinctly
    fig.add_trace(go.Scatter(
        x=[nodes_df.loc[home_node_id, 'x']],
        y=[nodes_df.loc[home_node_id, 'y']],
        mode='markers',
        marker=dict(symbol='x', size=12, color='red'),
        name='Home Node',
        hoverinfo='text',
        hovertext=f"Home: Node {home_node_id}"
    ))

    # Update the plot layout
    fig.update_layout(
        title="Solution Route Visualization",
        xaxis=dict(title="X Coordinate"),
        yaxis=dict(title="Y Coordinate", scaleanchor="x", scaleratio=1),
        legend=dict(title="Legend"),
        plot_bgcolor='white',
        hovermode='closest',
        height=500,
        width=700
    )

    # Show the figure
    fig.show()

def plot_refined_clusters(refined_clusters, nodes_df, home_node_id=0):
    """Plot all nodes in `refined_clusters` with distinct color schemes and highlight the home node distinctly."""
    # Initialize Plotly figure
    fig = go.Figure()

    # Color palette for different clusters (you can expand or modify these)
    colors = ["#76B041", "#F8766D", "#00BFC4", "#C77CFF", "#FF61CC", "#FFB400", "#FF0066", "#007FFF"]

    # Helper function to add nodes from a cluster to the plot
    def add_nodes_to_plot(cluster_label, node_indices, color):
        x = [nodes_df.loc[node_id, 'x'] for node_id in node_indices]
        y = [nodes_df.loc[node_id, 'y'] for node_id in node_indices]

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color=color, size=10),
            name=str(cluster_label),
            hoverinfo='text',
            hovertext=[
                f"Node: {node_id}<br>Days: {list(nodes_df.loc[node_id, 'opening_hours'].keys())}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f} <br>Angle to home: {nodes_df.loc[node_id, 'angle_to_home']} <br>Distance to home: {nodes_df.loc[node_id, 'dist_to_home']}"
                for node_id in node_indices
            ]
        ))

    # Add nodes from each refined cluster to the plot
    for index, (cluster_label, node_indices) in enumerate(refined_clusters.items()):
        color = colors[index % len(colors)]  # Choose a color based on the current index
        add_nodes_to_plot(cluster_label, node_indices, color)

    # Identify all nodes that are in any cluster
    assigned_nodes = {node for nodes in refined_clusters.values() for node in nodes}

    # Identify unassigned nodes
    all_node_ids = set(nodes_df.index)
    unassigned_nodes = all_node_ids - assigned_nodes - {home_node_id}

    # Add unassigned nodes to the plot
    if unassigned_nodes:
        fig.add_trace(go.Scatter(
            x=[nodes_df.loc[node_id, 'x'] for node_id in unassigned_nodes],
            y=[nodes_df.loc[node_id, 'y'] for node_id in unassigned_nodes],
            mode='markers',
            marker=dict(color='#B0B0B0', size=5),
            name='Unassigned Nodes',
            hoverinfo='text',
            hovertext=[
                f"Node: {node_id}<br>Days: {list(nodes_df.loc[node_id, 'opening_hours'].keys())}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f} <br>Angle to home: {nodes_df.loc[node_id, 'angle_to_home']} <br>Distance to home: {nodes_df.loc[node_id, 'dist_to_home']}"
                for node_id in unassigned_nodes
            ]
        ))

    # Add the home node distinctly
    fig.add_trace(go.Scatter(
        x=[nodes_df.loc[home_node_id, 'x']],
        y=[nodes_df.loc[home_node_id, 'y']],
        mode='markers',
        marker=dict(symbol='x', size=12, color='black'),
        name='Home Node',
        hoverinfo='text',
        hovertext=f"Home: Node {home_node_id}"
    ))

    # Update the plot layout
    fig.update_layout(
        title="Node Allocation by Refined Clusters",
        xaxis=dict(title="X Coordinate", range=[-5, 105]),
        yaxis=dict(title="Y Coordinate", range=[-5, 105], scaleanchor="x", scaleratio=1),
        legend=dict(title="Clusters", itemsizing='constant'),
        plot_bgcolor='white',
        hovermode='closest',
        height=500,
        width=700
    )

    # Show the figure
    fig.show()

def plot_all_cluster_routes(route_lists, nodes_df, home_node_id=0):
    """Plot routes for all clusters with distinct color schemes and highlight the home node distinctly."""
    # Initialize Plotly figure
    fig = go.Figure()

    # Color palette for different clusters
    colors = ["#76B041", "#F8766D", "#00BFC4", "#C77CFF", "#FF61CC", "#FFB400", "#FF0066", "#007FFF"]

    # Plot each cluster's route
    for index, (cluster_label, route_list) in enumerate(route_lists.items()):
        color = colors[index % len(colors)]  # Choose a color based on the current index

        # Extract node IDs and times for plotting coordinates and hover text
        node_ids = [node_id for node_id, _ in route_list]  # Extract just the node IDs from the tuples
        times = [time for _, time in route_list]  # Extract times for hover text
        x = [nodes_df.loc[node_id, 'x'] for node_id in node_ids]
        y = [nodes_df.loc[node_id, 'y'] for node_id in node_ids]

        # Adding the route trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(
                color=[
                    'black' if isinstance(nodes_df.loc[node_id, 'fixed_appointment'], list) else color
                    for node_id in node_ids
                ],
                size=8
            ),
            name=f"Route {cluster_label}",
            hoverinfo='text',
            hovertext=[
                f"Node: {node_id}<br> \
                Time: {time}<br>\
                Days: {list(nodes_df.loc[node_id, 'opening_hours'].keys())}<br>\
                Priority: {nodes_df.loc[node_id, 'priority']:.2f}<br>\
                Angle to home: {nodes_df.loc[node_id, 'angle_to_home']}<br>\
                Distance to home: {nodes_df.loc[node_id, 'dist_to_home']}<br>\
                Time Window: {nodes_df.loc[node_id, 'adjusted_opening_hours']}<br>\
                "
                for node_id, time in zip(node_ids, times)
            ]
        ))

    # Add the home node distinctly
    home_x = nodes_df.loc[home_node_id, 'x']
    home_y = nodes_df.loc[home_node_id, 'y']
    fig.add_trace(go.Scatter(
        x=[home_x],
        y=[home_y],
        mode='markers',
        marker=dict(symbol='star', size=15, color='red'),
        name='Home Node',
        hoverinfo='text',
        hovertext=f"Home: Node {home_node_id}<br>X: {home_x}<br>Y: {home_y}"
    ))

    # Update the plot layout
    fig.update_layout(
        title="All Cluster Routes Visualization",
        xaxis=dict(title="X Coordinate"),
        yaxis=dict(title="Y Coordinate", scaleanchor="x", scaleratio=1),
        legend=dict(title="Clusters & Routes", itemsizing='constant'),
        plot_bgcolor='white',
        hovermode='closest',
        height=600,
        width=800
    )

    # Show the figure
    fig.show()

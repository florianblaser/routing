import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from scipy.spatial import distance
import plotly.graph_objs as go

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_nodes_dataframe(num_nodes, home_node_id, min_work_days, visiting_interval_min=10, visiting_interval_max=30, max_last_visit=30):
    """Create a DataFrame containing nodes' information, including their attributes and distances."""
    node_ids = np.arange(0, num_nodes)
    
    def generate_opening_hours():
        """Generate opening hours for a random subset of days."""
        schedule = {}
        # Randomly select a minimal number of working days within a given week
        days = sorted(random.sample(range(5), min_work_days))
        for day in days:
            # Generate random start hours and end hours within constraints
            start_hour = 8 + random.randint(0, 6)
            end_hour = start_hour + random.randint(1, 6) * 2
            if end_hour > 20:
                end_hour = 20
            schedule[day] = (f"{start_hour:02d}:00", f"{end_hour:02d}:00")
        return schedule
    
    def convert_to_time(hours_dict):
        """Convert string-based opening hours to datetime.time objects."""
        return {k: (datetime.strptime(v[0], "%H:%M").time(), datetime.strptime(v[1], "%H:%M").time()) for k, v in hours_dict.items()}
    
    current_date = datetime.now()
    last_visited = [current_date - timedelta(days=np.random.randint(1, max_last_visit)) for _ in range(num_nodes)]
    last_visited = [date.strftime("%Y-%m-%d") for date in last_visited]
    visiting_intervals = np.random.randint(visiting_interval_min, visiting_interval_max, size=num_nodes)
    durations = np.random.randint(30, 61, size=num_nodes)

    # Consolidate various attributes into a DataFrame for each node
    nodes_df = pd.DataFrame({
        "Node_ID": node_ids,
        "Opening_Hours": [generate_opening_hours() for _ in range(num_nodes)],
        "last_visited": last_visited,
        "Visiting Interval (days)": visiting_intervals,
        "on_site_time": durations
    })

    nodes_df.sort_values("Node_ID", inplace=True)
    nodes_df['days_since_last_visit'] = (current_date - pd.to_datetime(nodes_df['last_visited'])).dt.days
    nodes_df['priority'] = nodes_df['days_since_last_visit'] / nodes_df['Visiting Interval (days)']
    nodes_df['priority'] = nodes_df['priority'].apply(lambda x: 1 if x > 1 else x)

    nodes_df['Opening_Hours'] = nodes_df['Opening_Hours'].apply(convert_to_time)

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

    # Create a distance matrix based on Euclidean distance
    def calculate_euclidean_distance_matrix(coords):
        return distance.cdist(coords, coords, 'euclidean')

    distance_matrix = calculate_euclidean_distance_matrix(nodes_df[['x', 'y']].values)
    distance_df = pd.DataFrame(np.round(distance_matrix, 2), columns=range(0, num_nodes), index=range(0, num_nodes))

    # Ensure the diagonal is zero to represent self-distance
    np.fill_diagonal(distance_df.values, 0)

    return nodes_df, distance_df
 
def calculate_metric(cluster_indices, nodes_df):
    """
    Calculates the metric for a cluster.
    The metric is the sum of priorities and the number of nodes.
    """
    cluster_data = nodes_df.loc[cluster_indices]
    sum_priorities = cluster_data['priority'].sum()
    num_nodes = len(cluster_indices)
    return num_nodes # + sum_priorities

def adjust_angles(clusters, nodes_df, angle_ranges):
    sorted_nodes = nodes_df.sort_values(by='angle_to_home').reset_index()
    gaps = [(sorted_nodes['angle_to_home'][i + 1] - sorted_nodes['angle_to_home'][i], i, i + 1) for i in range(len(sorted_nodes) - 1)]
    gaps.append((360 - sorted_nodes['angle_to_home'].iloc[-1] + sorted_nodes['angle_to_home'].iloc[0], len(sorted_nodes) - 1, 0))
    max_gap, index1, index2 = max(gaps, key=lambda x: x[0])
    if max_gap > 90:
        current_angle = sorted_nodes.iloc[index1]['angle_to_home']
        end_angle = sorted_nodes.iloc[index2]['angle_to_home']

    else:
        current_angle = 0
        end_angle = 360
    
    total_nodes = len(nodes_df)
    num_clusters = len(clusters)
    desired_size = total_nodes / num_clusters

    # Calculate current sizes and prepare adjustments
    current_sizes = {cluster_id: len(indices) for cluster_id, indices in clusters.items()}
    angle_changes = {}

    # Calculate total angle change needed based on node distribution
    total_angle_change = 0
    for cluster_id, size in current_sizes.items():
        deviation = size - desired_size
        # Here, angle adjustment could be proportional to the deviation from desired size
        angle_changes[cluster_id] = -deviation * 1  # Adjust scaling factor as needed
        total_angle_change += angle_changes[cluster_id]

    # Ensure total angle remains 360 degrees
    correction = -total_angle_change / num_clusters

    # Adjust angles
    adjusted_angle_ranges = {}
    current_angle = 0
    for cluster_id, initial_range in angle_ranges.items():
        start_angle, end_angle = initial_range
        angle_width = end_angle - start_angle + angle_changes[cluster_id] + correction
        adjusted_angle_ranges[cluster_id] = (current_angle, current_angle + angle_width)
        current_angle += angle_width
    return adjusted_angle_ranges

def custom_clustering(distance_matrix, nodes_df, num_small_clusters, num_large_clusters, overnight_factor, precision, adjustment_speed, home_node_id = 0):
    """
    Custom clustering algorithm that divides nodes into angular clusters.
    Balances clusters between small and large based on the metric calculations.
    Includes the home node at the start of each cluster.
    
    Args:
    - distance_matrix: DataFrame containing distances between nodes.
    - nodes_df: DataFrame containing node data.
    - num_small_clusters: Number of smaller clusters to create.
    - num_large_clusters: Number of larger clusters to create.
    - factor: Adjustment factor used in balancing clusters.
    - precision: Number of iterations for refining the clusters.
    - home_node_id: ID of the home node to be included at the start of each cluster.
    """
    # remove home node from nodes_df
    nodes_df = nodes_df.drop(home_node_id)
    # Initialize clusters
    clusters = {f'small_{i}': [home_node_id] for i in range(num_small_clusters)}
    clusters.update({f'large_{i}': [home_node_id] for i in range(num_large_clusters)})

    # Initial angle ranges equally distributed
    num_clusters = num_small_clusters + num_large_clusters
    step = 360 / num_clusters
    angle_ranges = {f'small_{i}': (i * step, (i + 1) * step) for i in range(num_small_clusters)}
    angle_ranges.update({f'large_{i}': ((i + num_small_clusters) * step, (i + num_small_clusters + 1) * step) for i in range(num_large_clusters)})

    # Iteratively adjust cluster assignments based on the metrics
    for _ in range(precision):
        # Start each iteration by clearing clusters but keeping the home node
        for key in clusters.keys():
            clusters[key] = [home_node_id]

        # Distribute nodes based on (adjusted) angle ranges
        for index, row in nodes_df.iterrows():
            if index == home_node_id:
                continue  # Skip the home node in regular assignments
            node_angle = row['angle_to_home']
            for cluster_id, (start_angle, end_angle) in angle_ranges.items():
                if start_angle <= node_angle < end_angle:
                    clusters[cluster_id].append(index)
                    break

        # Adjust angles based on cluster metrics
        angle_ranges = adjust_angles(clusters, nodes_df, angle_ranges) #, adjustment_speed, overnight_factor, num_small_clusters, num_large_clusters)

    return clusters

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
        sub_nodes_df = nodes_df[nodes_df['Node_ID'].isin(nodes)].reset_index(drop=True)
        sub_distance_matrix = distance_matrix.loc[nodes, nodes].values.tolist()
        sub_distance_matrix = [[int(x) for x in row] for row in sub_distance_matrix]
        node_mapping = sub_nodes_df['Node_ID'].to_dict()
        
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
            name=cluster_label,
            hoverinfo='text',
            hovertext=[
                f"Node: {node_id}<br>Days: {list(nodes_df.loc[node_id, 'Opening_Hours'].keys())}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f} <br>Angle to home: {nodes_df.loc[node_id, 'angle_to_home']}"
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
                f"Node: {node_id}<br>Days: {list(nodes_df.loc[node_id, 'Opening_Hours'].keys())}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f} <br>Angle to home: {nodes_df.loc[node_id, 'angle_to_home']}"
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

        # Coordinates for the route
        x = [nodes_df.loc[node_id, 'x'] for node_id in route_list]
        y = [nodes_df.loc[node_id, 'y'] for node_id in route_list]

        # Adding the route trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8),
            name=f"Route {cluster_label}",
            hoverinfo='text',
            hovertext=[
                f"Node: {node_id}<br>Days: {list(nodes_df.loc[node_id, 'Opening_Hours'].keys())}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f}"
                for node_id in route_list
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
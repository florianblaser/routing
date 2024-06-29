import bisect
import math
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime, timedelta, time
import random
from collections import Counter
import plotly.graph_objects as go
from scipy.spatial import distance

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# VISUALIZATION
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
    fig = go.Figure()
    colors = ["#76B041", "#F8766D", "#00BFC4", "#C77CFF", "#FF61CC", "#FFB400", "#FF0066", "#007FFF"]

    def add_nodes_to_plot(cluster_label, node_indices, color):
        x = [nodes_df.loc[node_id, 'x'] for node_id in node_indices]
        y = [nodes_df.loc[node_id, 'y'] for node_id in node_indices]

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color=color, size=8),
            name=str(cluster_label),
            hoverinfo='text',
            hovertext=[
                f"Node: {node_id}<br>Days: {list(nodes_df.loc[node_id, 'opening_hours'].keys())}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f} <br>Angle to home: {nodes_df.loc[node_id, 'angle_to_home']} <br>Distance to home: {nodes_df.loc[node_id, 'dist_to_home']} <br>On site time: {nodes_df.loc[node_id, 'on_site_time']}<br>"
                for node_id in node_indices
            ]
        ))
        
        # Plot a smaller black dot for nodes with 'fixed_appointment'
        fixed_appointment_indices = [node_id for node_id in node_indices if isinstance(nodes_df.loc[node_id, 'fixed_appointment'], list)]
        if fixed_appointment_indices:
            fx = [nodes_df.loc[node_id, 'x'] for node_id in fixed_appointment_indices]
            fy = [nodes_df.loc[node_id, 'y'] for node_id in fixed_appointment_indices]

            fig.add_trace(go.Scatter(
                x=fx,
                y=fy,
                mode='markers',
                marker=dict(color='black', size=4),  # Smaller size for the black dot
                name=f'Fixed {cluster_label}'
            ))

    for index, (cluster_label, node_indices) in enumerate(refined_clusters.items()):
        color = colors[index % len(colors)]
        add_nodes_to_plot(cluster_label, node_indices, color)

    assigned_nodes = {node for nodes in refined_clusters.values() for node in nodes}
    all_node_ids = set(nodes_df.index)
    unassigned_nodes = all_node_ids - assigned_nodes - {home_node_id}

    if unassigned_nodes:
        fig.add_trace(go.Scatter(
            x=[nodes_df.loc[node_id, 'x'] for node_id in unassigned_nodes],
            y=[nodes_df.loc[node_id, 'y'] for node_id in unassigned_nodes],
            mode='markers',
            marker=dict(color='#B0B0B0', size=5),
            name='Unassigned Nodes'
        ))

    fig.add_trace(go.Scatter(
        x=[nodes_df.loc[home_node_id, 'x']],
        y=[nodes_df.loc[home_node_id, 'y']],
        mode='markers',
        marker=dict(symbol='x', size=12, color='black'),
        name='Home Node'
    ))

    fig.update_layout(
        title="Node Allocation by Refined Clusters",
        xaxis=dict(title="X Coordinate"),
        yaxis=dict(title="Y Coordinate", scaleanchor="x", scaleratio=1),
        legend=dict(title="Clusters", itemsizing='constant'),
        plot_bgcolor='white',
        hovermode='closest',
        height=500,
        width=700
    )

    fig.show()
    
def plot_all_cluster_routes(route_lists, nodes_df, overnight_nodes, home_node_id=0):
    """Plot routes for all clusters with distinct color schemes and highlight the home node and nodes with fixed appointments distinctly, and overnight nodes with an 'X' marker."""
    # Initialize Plotly figure
    fig = go.Figure()

    # Color palette for different clusters
    colors = ["#76B041", "#F8766D", "#00BFC4", "#C77CFF", "#FF61CC", "#FFB400", "#FF0066", "#007FFF"]

    # Plot each cluster's route
    for index, (cluster_label, route_list) in enumerate(route_lists.items()):
        match_cluster_label = cluster_label[:cluster_label.find('_{')]
        color = colors[index % len(colors)]  # Choose a color based on the current index

        # Extract node IDs for plotting coordinates
        visited_node_ids = [node_id for node_id, _ in route_list]
        x_visited = [nodes_df.loc[node_id, 'x'] for node_id in visited_node_ids]
        y_visited = [nodes_df.loc[node_id, 'y'] for node_id in visited_node_ids]

        # Determine marker properties: check if the node is an overnight node
        if overnight_nodes:
            marker_symbols = ['x' if node_id in overnight_nodes else 'circle' for node_id in visited_node_ids]
        else:
            marker_symbols = 'circle'
        marker_colors_visited = [
            'black' if isinstance(nodes_df.loc[node_id, 'fixed_appointment'], list) else color
            for node_id in visited_node_ids
        ]

        # Hover text for visited nodes
        hover_text_visited = [
            f"Node: {node_id}<br>Time: {time}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f}<br>"
            f"Days: {list(nodes_df.loc[node_id, 'opening_hours'].keys())}<br>"
            f"Time Window: {nodes_df.loc[node_id, 'adjusted_opening_hours_hhmm']}<br>"
            f"On site time: {nodes_df.loc[node_id, 'on_site_time']}<br>"
            for node_id, time in route_list
        ]

        # Adding the route trace for visited nodes
        fig.add_trace(go.Scatter(
            x=x_visited,
            y=y_visited,
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(symbol=marker_symbols, color=marker_colors_visited, size=10),
            name=f"Visited - {cluster_label}",
            hoverinfo='text',
            hovertext=hover_text_visited
        ))

        # Identify unvisited nodes in this cluster
        unvisited_nodes = nodes_df[(nodes_df['cluster_name'] == match_cluster_label) & (~nodes_df['node_id'].isin(visited_node_ids))]
        if not unvisited_nodes.empty:
            x_unvisited = unvisited_nodes['x']
            y_unvisited = unvisited_nodes['y']

            # Determine marker color for unvisited nodes
            marker_colors_unvisited = [
                'black' if isinstance(nodes_df.loc[node_id, 'fixed_appointment'], list) else  color
                for node_id in unvisited_nodes['node_id']
            ]

            # Hover text for unvisited nodes
            hover_text_unvisited = [
                f"Node: {node_id}<br>Priority: {nodes_df.loc[node_id, 'priority']:.2f}<br>"
                f"Days: {list(nodes_df.loc[node_id, 'opening_hours'].keys())}<br>"
                f"Time Window: {nodes_df.loc[node_id, 'adjusted_opening_hours_hhmm']}<br>"
                f"On site time: {nodes_df.loc[node_id, 'on_site_time']}<br>"
                for node_id in unvisited_nodes['node_id']
            ]

            # Adding trace for unvisited nodes
            fig.add_trace(go.Scatter(
                x=x_unvisited,
                y=y_unvisited,
                mode='markers',
                marker=dict(color=marker_colors_unvisited, size=6, opacity=0.5),
                name=f"Unvisited - {cluster_label}",
                hoverinfo='text',
                hovertext=hover_text_unvisited
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
    return fig

# DATA GENERATION
def create_data(num_nodes, frac_fixed_app, home_node_id, min_work_days, days_off, visiting_interval_min=10, visiting_interval_max=30, max_last_visit=30, simple_schedule=False):
    """Create a DataFrame containing nodes' information, including their attributes and distances."""
    node_ids = np.arange(0, num_nodes)
    
    def generate_opening_hours(min_work_days):
        """Generate opening hours for a random subset of days."""
        schedule = {}
        # Randomly select a minimal number of working days within a given week
        days = sorted(random.sample(range(1, 8), min_work_days))
        for day in days:
            # Generate random start hours and end hours within constraints
            start_hour = 8 + random.randint(0, 1) # Start between 8:XX and 9:XX
            end_hour = start_hour + random.randint(9, 10) # End between 17:XX and 19:XX
            start_minute = random.randint(0, 59)
            end_minute = random.randint(0, 59)
            start_minute = 0
            end_minute = 0

            # Randomly add lunch break
            if random.random() < 0.1:
                lunch_start = 12 + random.randint(0, 1)
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
    
    def add_fixed_appointments(nodes_df, frac_fixed_app):
        selected_indices = nodes_df.sample(frac=frac_fixed_app, replace=True).index
        for idx in selected_indices:
            # day can't be in days_off
            try:
                day = random.choice([day for day in nodes_df.at[idx, 'opening_hours'].keys() if day not in days_off])
            except:
                continue
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
    
    nodes_df['weeks_until_visit'] = (nodes_df['Visiting Interval (days)'] - nodes_df['days_since_last_visit']) / 7

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
    nodes_df['weekday_fixed_appointment'] = nodes_df['fixed_appointment'].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    
    # PRIORITY
    bins = [-float('inf'), 1, 2, 3, float('inf')]
    nodes_df['priority'] = pd.cut(nodes_df['weeks_until_visit'], bins=bins, labels=[1000, 500, 250, 100], right=False).astype(int)
    nodes_df.loc[~nodes_df['weekday_fixed_appointment'].isna(), 'priority'] = 5000
    # nodes_df.loc[~nodes_df['weekday_fixed_appointment'].isna(), 'priority'] = 2
    # nodes_df['priority'] = (nodes_df['priority']*100)**penalty_factor[0]
    # nodes_df['priority'] = round((nodes_df['priority'] - nodes_df['priority'].min()) / (nodes_df['priority'].max() - nodes_df['priority'].min()) * penalty_factor[1] + penalty_factor[2], 0)
    # nodes_df['priority'] = nodes_df['priority'].astype(int)

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

    nodes_df.loc[nodes_df['node_id'] == 0, 'opening_hours'] = [{
        1: [[time(0, 1), time(23, 59)]],
        2: [[time(0, 1), time(23, 59)]],
        3: [[time(0, 1), time(23, 59)]],
        4: [[time(0, 1), time(23, 59)]],
        5: [[time(0, 1), time(23, 59)]],
        6: [[time(0, 1), time(23, 59)]],
        7: [[time(0, 1), time(23, 59)]]
    }] * len(nodes_df[nodes_df['node_id'] == 0])
    nodes_df.loc[nodes_df['node_id'] == 0, 'closed_days'] = [{}] * len(nodes_df[nodes_df['node_id'] == 0])
    nodes_df.loc[nodes_df['node_id'] == 0, 'fixed_appointment'] = [np.nan] * len(nodes_df[nodes_df['node_id'] == 0])
    nodes_df.loc[nodes_df['node_id'] == 0, 'weekday_fixed_appointment'] = [np.nan] * len(nodes_df[nodes_df['node_id'] == 0])
    return nodes_df, distance_df
 
 # CLUSTERING

# DATA ENGINEERING
def fit_blocks(unique_cluster_blocks, gaps, solution, solutions, index=0, verbose=False):
    if index == len(gaps):  # Check if we've addressed all gaps
        if not unique_cluster_blocks:  # Ensure all blocks have been used
            solutions.append(solution)
        return
    
    # Attempt to fit each block into the current gap
    for i, block in enumerate(unique_cluster_blocks):
        try:
            size = block.split('_')[0]
        except:
            print(f"WARNING: {block}")
        block_size = float(size)
        if verbose:
            print(f"gaps in iteration {index}:", gaps)
        if block_size <= float(gaps[index]):  # Check if block can fit in the current gap
            # Setup for recursion: remove the block and reduce the gap size
            if verbose:
                print(f'fitting {block} of size {block_size} into gap of size {gaps[index]}')
            new_blocks = unique_cluster_blocks[:i] + unique_cluster_blocks[i+1:]  # Remove current block
            new_solution = [lst[:] for lst in solution]  # Copy solution to modify
            new_solution[index].append(block)  # Add block to current gap's solution
            if verbose:
                print(f'new solution: {new_solution}')
            
            # Reduce the gap by the size of the block
            if verbose:
                print('gaps before:', gaps)
            new_gaps = gaps[:]
            new_gaps[index] -= block_size
            if verbose:
                print('gaps after:', new_gaps)
                print(f'new gap size: {new_gaps[index]}')
            
            # Move to next gap if current gap is exactly filled, otherwise continue
            if new_gaps[index] == 0.0:
                fit_blocks(new_blocks, new_gaps, new_solution, solutions, index + 1)
            else:
                if verbose:
                    print(f'Continuing with gap of size {new_gaps[index]}')
                fit_blocks(new_blocks, new_gaps, new_solution, solutions, index)
        else:
            if verbose:
                print(f'Block {block} of size {block_size} does not fit in gap of size {gaps[index]}')

def get_options_df(days_off, relative_day_lengths, no_overnight_stays, max_days_off):
    def find_lists(current_list, current_sum, max_length, target_sum):
        if current_sum > target_sum or len(current_list) > max_length:
            return []
        if current_sum == target_sum and len(current_list) <= max_length:
            return [current_list]
        results = []
        for i in range(0, 8):
            adjusted_sum = current_sum + (i if i > 0 else 1)  # Adjust sum for 0s treated as 1
            results.extend(find_lists(current_list + [i], adjusted_sum, max_length, target_sum))
        return results

    # Generate valid lists
    valid_lists = find_lists([], 0, 7, 7)

    # Function to repeat values in the list according to their integer value
    def repeat_values(lst, overnight_trips=0):
        repeated_list = []
        for num in lst:
            if overnight_trips == 1:
                repeated_list.extend([num] * (num - 1) + [0] if num > 0 else [0])
            else:
                repeated_list.extend([num] * num if num > 0 else [0])
        return repeated_list

    # Calculate days off for each list
    def calculate_trips(lst, trip):
        repeated_lst = repeat_values(lst)
        days_off = []
        for i in range(min(len(repeated_lst), 7)):
            if repeated_lst[i] == trip:
                days_off.append(i + 1)  # Use 1-based indexing for days
        return days_off

    # Calculate trip days for each list
    def calculate_overnight_trips(lst):
        repeated_lst = repeat_values(lst, overnight_trips=1)  
        trip_days = []
        for i in range(min(len(repeated_lst), 7)):
            if repeated_lst[i] > 1:
                trip_days.append(i + 1)  # Use 1-based indexing for days
        return trip_days

    def calculate_overnight_stays(lst):
        overnight_stays = 0
        for i in range(min(len(lst), 7)):
            if lst[i] > 1:
                overnight_stays += lst[i] - 1
        return overnight_stays

    def combine_day_lengths(day_lengths, multipliers):
        combined_lengths = []
        i = 0
        multiplier_idx = 0

        while i < len(day_lengths) and multiplier_idx < len(multipliers):
            if multipliers[multiplier_idx] > 1:
                combined_length = sum(day_lengths[i:i + multipliers[multiplier_idx]])
                combined_lengths.append(combined_length)
                i += multipliers[multiplier_idx]
            else:
                combined_lengths.append(day_lengths[i])
                i += 1
            multiplier_idx += 1
        
        # Ensure the combined lengths list is of the correct length
        while len(combined_lengths) < len(multipliers):
            combined_lengths.append(0.0)
        
        return combined_lengths
    
    data = {
        'gaps': valid_lists,
        'Sum': [sum(lst) for lst in valid_lists],  # Calculating sum normally, 0s count as 0
        'Length': [len(lst) for lst in valid_lists],
        'n_overnight_trips': [calculate_overnight_stays(lst) for lst in valid_lists],
        'overnight_days': [calculate_overnight_trips(lst) for lst in valid_lists],
        'off_days': [calculate_trips(lst, 0) for lst in valid_lists],
        'n_days_off': [lst.count(0) for lst in valid_lists], 
    }

    options_df = pd.DataFrame(data)

    # Filter the DataFrame based on the constraints
    options_df = options_df[options_df['off_days'].apply(lambda x: all(day in x for day in days_off))]
    options_df = options_df[options_df['n_days_off'].apply(lambda x: x <= max_days_off)]
    options_df = options_df[options_df['overnight_days'].apply(lambda x: not any(day in x for day in no_overnight_stays))]
    options_df['blocks'] = options_df['gaps'].apply(lambda x: Counter([item for item in x if item > 0]))
    options_df['gaps_cont'] = options_df['gaps'].apply(lambda x: combine_day_lengths(relative_day_lengths, x))
    options_df['blocks_cont'] = options_df['gaps_cont'].apply(lambda x: Counter([item for item in x if item > 0]))
    options_df['total_trips'] = options_df['blocks'].apply(lambda x: sum(x.values()))
    options_df = options_df.sort_values(by='n_overnight_trips')
    options_df = options_df.reset_index(drop=True)
    return options_df

def adjust_opening_hours(row, work_schedule, margin, time_matrix):
    clusters = row['cluster']
    first_day = min(clusters)
    opening_hours = row['opening_hours']
    fixed_appointment = row['fixed_appointment']
    on_site_time = row['on_site_time']
    adjusted_hours = []
    
    if isinstance(fixed_appointment, list):
        day, app_start, _ = fixed_appointment[0], fixed_appointment[1], fixed_appointment[2]
        adjusted_open = time_to_minutes(app_start) + 1440 * (day - first_day) - margin 
        adjusted_close = adjusted_open
        adjusted_hours = [[adjusted_open, adjusted_close]]
    else:
        for day, intervals in opening_hours.items():
            if day in clusters:  # Ensure we only adjust days that are in clusters
                for start, end in intervals:
                    work_start, work_end = work_schedule[day]
                    start = max(work_start, time_to_minutes(start))
                    end = min(work_end, time_to_minutes(end))

                    # Adjust each interval's start and end times
                    adjusted_start = start + 1440 * (day - first_day)
                    distance_to_home = time_matrix.loc[row['node_id'], 0]
                    adjusted_end = end + 1440 * (day - first_day) - on_site_time - distance_to_home # needs to finish before store closes, work ends and consider the travel time to home
                    adjusted_end = int(round(adjusted_end, 0))

                    adjusted_hours.append([adjusted_start, adjusted_end])
    return adjusted_hours

# CLUSTERING
def custom_clustering(nodes_df, time_matrix, blocks_cont, precision=100, home_node_id=0, verbose=False, visual=False):
    def calculate_metric(nodes_df, time_matrix, global_max_dist, node_ids, cluster_id, print_ind_metrics=False):
        if len(node_ids) > 1:
            total_spread = np.mean(time_matrix)
            filtered_nodes_df = nodes_df[nodes_df['node_id'].isin(node_ids)]

            num_nodes_metric = len(filtered_nodes_df) / len(nodes_df)
            num_fixed_metric = len(filtered_nodes_df[filtered_nodes_df['fixed_appointment'].apply(lambda x: isinstance(x, list))]) / len(filtered_nodes_df)

            filtered_time_matrix = time_matrix.iloc[node_ids,node_ids]
            spread_metric = np.mean(filtered_time_matrix) / total_spread
            
            priority_metric = filtered_nodes_df['priority'].nlargest(int(0.5 * len(filtered_nodes_df))).mean()
            priority_metric = priority_metric / float(cluster_id.split('_')[0])

            max_dist_to_root = filtered_nodes_df['dist_to_home'].max()
            dist_metric = max_dist_to_root / global_max_dist

            # prevent any metric from being nan
            if np.isnan(num_nodes_metric):
                # print(f'Problems with num_nodes_metric for {cluster_id}')
                num_nodes_metric = 0.5
            if np.isnan(priority_metric):
                # print(f'Problems with priority_metric for {cluster_id}')
                priority_metric = 0.3
            if np.isnan(dist_metric):
                # print(f'Problems with dist_metric for {cluster_id}')
                dist_metric = 0.5

            metric = num_nodes_metric * 2 + spread_metric # + num_fixed_metric/3

            if verbose==2:
                print(f'Cluster: {cluster_id}')
                print(f"Number of nodes metric: {round(num_nodes_metric, 2)}")
                print(f"Number of fixed appointments metric: {round(num_fixed_metric, 2)}")
                # print(f"Priority metric: {priority_metric}")
                # print(f"Distance metric: {dist_metric}")
                print(f"Spread metric: {spread_metric.round(2)}")
                print(f"Overall metric: {metric.round(2)}")
        else:
            metric = 0

            if print_ind_metrics:
                print('Found a cluster wihtout nodes')
        
        return metric

    def adjust_angles(clusters, nodes_df, time_matrix, angle_sizes, degree_adj, global_max_dist, blocks_cont, total_span, verbose):
        metrics = {}
        for cluster_id, node_ids in clusters.items():
            metrics[cluster_id] = calculate_metric(nodes_df, time_matrix, global_max_dist, node_ids, cluster_id)

        total_metric = sum(metrics.values())
        # total_days = sum of each key multiplied by the value in blocks
        total_days = sum([key * value for key, value in blocks_cont.items()])
        base_metric = total_metric / total_days
        
        target_metrics = {}
        for cluster, metric in metrics.items():
            size = float(cluster.split('_')[0])
            target_metrics[cluster] = base_metric * size

        new_angle_sizes = angle_sizes.copy()  # Copy existing angle sizes to modify
        
        deviations = {}
        for cluster_id, metric in metrics.items():
            size = float(cluster_id.split('_')[0])
            soll_metric = target_metrics[cluster_id]
            deviation = metric - soll_metric
            deviations[cluster_id] = deviation
            new_angle_sizes[cluster_id] -= deviation * degree_adj

        # Normalize the new angles to ensure they sum to total_span
        total_new_angles = sum(new_angle_sizes.values())
        scale_factor = total_span / total_new_angles
        for cluster_id in new_angle_sizes:
            new_angle_sizes[cluster_id] *= scale_factor

        if verbose == 2:
            print("Deviations, metrics, and new angle sizes:")
            for cluster_id in clusters:
                print(f"Cluster {cluster_id} with deviation {round(deviations[cluster_id], 2)}, "
                    f'and initial angle size {round(angle_sizes[cluster_id], 2)}° '
                    f"has new angle size {round(new_angle_sizes[cluster_id], 2)}°.")

        return new_angle_sizes

    # remove the home node from the nodes_df
    if nodes_df.index[0] == 0:
        nodes_df_copy = nodes_df.drop(0).copy()
    
    clusters = {}
    for size, count in blocks_cont.items():
        for i in range(count):
            clusters[f'{round(size, 2)}_day_trip_{i}'] = []
    
    angles = sorted(nodes_df_copy['angle_to_home'])
    diffs = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    diffs.append(360 - angles[-1] + angles[0])
    
    max_gap = max(diffs)
    gap_start = angles[diffs.index(max_gap)]
    gap_end = angles[(diffs.index(max_gap) + 1) % len(angles)]

    max_gap = max(diffs)
    total_span = 360 - max_gap

    if verbose == True:
        print(f"Largest gap spans from {gap_start}° to {gap_end}°, covering {max_gap}° leaving a total span of {total_span} for locations.")

    total_equivalent_degrees = sum(count * size for size, count in blocks_cont.items())
    base_degree = total_span / total_equivalent_degrees

    angle_sizes = {}
    for size, count in blocks_cont.items():
        # Calculate the angular size for each cluster of this size
        cluster_angle_size = base_degree * size
        for i in range(count):
            cluster_id = f'{round(size, 2)}_day_trip_{i}'
            angle_sizes[cluster_id] = cluster_angle_size
    global_max_dist = nodes_df_copy['dist_to_home'].max()

    cluster_start = gap_end
    degree_adj = total_span / 7

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

        if (i == 0 and verbose):
            print("Initial clusters:")
            for key, value in clusters.items():
                print(f"{key}: {value}")

        if i % 25 == 0 and visual:
            plot_refined_clusters(clusters, nodes_df)
        
        angle_sizes = adjust_angles(clusters, nodes_df_copy, time_matrix, angle_sizes, degree_adj, global_max_dist, blocks_cont, total_span, verbose)
        degree_adj *= 0.97

        sum_of_angles = sum(angle_sizes.values())
        if verbose == 2:
            print(f"Sum of angles: {sum_of_angles} vs. total span: {total_span}")
        if (i % 10 == 0) and verbose:
            for key in clusters.keys():
                print(key)
                print(f'Mean spread: {np.mean(time_matrix.iloc[clusters[key], clusters[key]].values)}')
                print(f'Number of nodes: {len(clusters[key])}')
    
    def circular_mean(angles):
        """Calculate the circular mean for given angles in degrees."""
        sin_sum = np.sum(np.sin(np.radians(angles)))
        cos_sum = np.sum(np.cos(np.radians(angles)))
        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    
    # final check that all nodes are assigned
    all_nodes = set(nodes_df.index)
    assigned_nodes = set([node for nodes in clusters.values() for node in nodes])
    unassigned_nodes = all_nodes - assigned_nodes
    if unassigned_nodes:
        
        mean_angles = {}
        for cluster_id, node_ids in clusters.items():
            angles = nodes_df.loc[node_ids, 'angle_to_home']
            mean_angle = circular_mean(angles)
            mean_angles[cluster_id] = mean_angle
        
        for node_id in unassigned_nodes:
            node_angle = nodes_df.loc[node_id, 'angle_to_home']
            closest_cluster = min(mean_angles, key=lambda x: abs((mean_angles[x] - node_angle + 180) % 360 - 180))
            clusters[closest_cluster].append(node_id)
        
        if verbose:
            print(f'node {node_id} with X {nodes_df.loc[node_id, "x"]} and Y {nodes_df.loc[node_id, "y"]} and angle {nodes_df.loc[node_id, "angle_to_home"]} was assigned to cluster {closest_cluster} with mean angle {mean_angles[closest_cluster]}')

    return clusters

# HELPER FUNCTIONS
def time_to_minutes(t):
    return t.hour * 60 + t.minute + t.second

def minutes_to_hhmm(minutes_am, days = False):
    if days:
        day = minutes_am // 1440
    minutes_am = minutes_am % 1440  # Ensure minutes are within a day
    minutes = minutes_am % 60
    hours = (minutes_am - minutes) // 60  # Use integer division for hours
    if days:
        return f'{hours:02}:{minutes:02} (day {day + 1})'
    else:
        return f'{hours:02}:{minutes:02}'

# ROUTING
def solve_vrp(time_matrix, sub_nodes_df, slack, lunch_duration, first_optimization_algorithm, second_optimization_algorithm, GlobalSpanCostCoefficient, work_schedule, time_limit, start_node, end_node=0, verbose=True):
    def create_data_model(sub_nodes_df, sub_time_matrix):
        """Stores the data for the problem."""
        data = {}
        data['time_matrix'] = sub_time_matrix
        data['windows'] = sub_nodes_df['adjusted_opening_hours'].tolist()
        data['priorities'] = sub_nodes_df['priority'].tolist()
        data['num_vehicles'] = 1
        data['on_site_time'] = sub_nodes_df['on_site_time'].tolist()
        data['start'] = start_node
        data['depot'] = end_node
        data['original_node_ids'] = sub_nodes_df['node_id'].tolist()
        return data

    def return_route_and_times(solution, manager, routing, original_node_ids, data):
        """Returns the route along with the start times at each node."""
        index = routing.Start(0)  # Start at the depot.
        route_with_travel = []
        route_without_travel = []
        # time_dimension = routing.GetDimensionOrDie('total_time')  # Make sure this matches the dimension name used

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            original_node_id = original_node_ids[node_index]  # Map back to original node ID
            time_var = time_dimension.CumulVar(index)
            start_time = solution.Min(time_var)
            end_time = start_time + data['on_site_time'][node_index]  # Include on-site time
            route_with_travel.append((original_node_id, minutes_to_hhmm(start_time, days=True), minutes_to_hhmm(end_time, days=True)))  # Include end time for better clarity
            route_without_travel.append((original_node_id, minutes_to_hhmm(start_time, days=True)))  # Include end time for better clarity
            next_index = solution.Value(routing.NextVar(index))
            
            travel_time = routing.GetArcCostForVehicle(index, next_index, 0) - data['on_site_time'][index]  # Get travel time
            route_with_travel.append(("road", travel_time))
            
            index = next_index

        # Add the final node
        final_node_index = manager.IndexToNode(index)
        final_node_id = original_node_ids[final_node_index]
        final_time_var = time_dimension.CumulVar(index)
        final_start_time = solution.Min(final_time_var)
        final_end_time = final_start_time + data['on_site_time'][final_node_index]
        route_with_travel.append((final_node_id, minutes_to_hhmm(final_start_time, days=True), minutes_to_hhmm(final_end_time, days=True)))
        route_without_travel.append((final_node_id, minutes_to_hhmm(final_start_time, days=True)))

        return route_with_travel, route_without_travel
    
    trip_name = sub_nodes_df['cluster_name'].values[0]
    trip_len = float(trip_name.split('_')[0])
    trip = sub_nodes_df['cluster'].values[0]
    first_day = min(trip)
    last_day = max(trip)
    _, global_work_end = work_schedule[last_day]
    global_work_end = global_work_end + 1440 * int(trip_len - 1)
    
    max_travel_time = int(5000 * trip_len)
    
    nodes = sub_nodes_df['node_id'].tolist()
    
    sub_time_matrix = time_matrix.loc[nodes, nodes].values.tolist()
    sub_time_matrix = [[int(x) for x in row] for row in sub_time_matrix]
    
    data = create_data_model(sub_nodes_df, sub_time_matrix)
    internal_start_index = data['original_node_ids'].index(data['start'])
    manager = pywrapcp.RoutingIndexManager(len(data["time_matrix"]), data["num_vehicles"], [internal_start_index], [data['depot']])
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node] + data['on_site_time'][from_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    routing.AddDimension(
        transit_callback_index,
        slack,  # upper bound for slack / waiting time
        max_travel_time,  # upper bound for vehicle maximum travel time
        False,  # start cumul to zero
        "total_time"
    )
    
    time_dimension = routing.GetDimensionOrDie("total_time")
    end_index = manager.NodeToIndex(internal_start_index)
    print(f'end index: {end_index} internal start index: {internal_start_index}, data depot: {data["depot"]}')
    time_dimension.CumulVar(end_index).SetRange(1, global_work_end)

    # PENALTY
    for location_index, priority in enumerate(data['priorities']):
        index = manager.IndexToNode(location_index)
        if index == 0:
            continue
        else:
            routing.AddDisjunction([location_index], int(priority))

    # OPENING HOURS, LUNCH AND OVERNIGHT BREAKS window example:
    for location_index, windows in enumerate(data['windows']):
        original_node_id = data['original_node_ids'][location_index]    
        print(f'location index: {location_index} and initial index {original_node_id}, windows: {windows}')
        if internal_start_index != 0:
            windows = [[1, 1439]]
        time_dimension.CumulVar(location_index).SetRange(windows[0][0], windows[-1][1])
        ranges = [[windows[i][1], windows[i+1][0]] for i in range(len(windows) - 1)]
        for start, end in ranges:
            time_dimension.CumulVar(index).RemoveInterval(start, end)
            if verbose == 4:
                print(f'removed interval {start} - {end} for window {windows}')
    
    # Agent lunch break (not sure why, if, how node_visit_transit is used or if it is even correct)
    node_visit_transit = {}
    for index in range(routing.Size()):
        node = manager.IndexToNode(index)
        node_visit_transit[index] = data['on_site_time'][node]
    
    lunch_breaks = []
    for day in range(first_day, last_day + 1):
        if trip_len >= 1:
            lunch_start = 12 * 60 + 1440 * (day - first_day)
            lunch_end = lunch_start + lunch_duration
            lunch_break_interval = routing.solver().FixedDurationIntervalVar(
                lunch_start, lunch_end, lunch_duration, False, f'lunch_break {day}'
            )
            lunch_breaks.append(lunch_break_interval)
            
            if verbose == 2:
                print(f'added lunch break from {lunch_start} to {lunch_end} for day {day} of {trip_name}')

    time_dimension.SetBreakIntervalsOfVehicle(lunch_breaks, 0, node_visit_transit)

    # Instantiate route start and end times to produce feasible times
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(0)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    strategy_mapping_first = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "BEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION
    }
    strategy_mapping_second = {
        "GREEDY_DESCENT": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    }
    search_parameters.first_solution_strategy = (strategy_mapping_first[first_optimization_algorithm])
    search_parameters.local_search_metaheuristic = (
            strategy_mapping_second[second_optimization_algorithm])
    
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = False

    # Optional: Set a more diverse objective to avoid synchronization
    time_dimension.SetGlobalSpanCostCoefficient(GlobalSpanCostCoefficient)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        dropped = []
        original_node_ids = sub_nodes_df['node_id'].tolist()
        for node in range(routing.Size()):
            if routing.IsStart(node) or routing.IsEnd(node):
                continue
            if solution.Value(routing.NextVar(node)) == node:
                dropped_node_id = original_node_ids[manager.IndexToNode(node)]
                dropped.append(dropped_node_id)
        return return_route_and_times(solution, manager, routing, original_node_ids, data), dropped, solution.ObjectiveValue()
    else:
        print(f"No solution found")
        return None

def find_last_nodes_per_day(trips):
    for trip, nodes in trips.items():
        # Check if the trip spans multiple days by counting the numbers in the curly brackets
        if len(trip.split('{')[1].split('}')[0].split(',')) > 1:
            # Initialize tracking variables
            current_day = None
            last_node = None
            day_nodes = []

            for node_id, time_stamp in nodes:
                # Extract day information from the timestamp
                day = time_stamp.split(' ')[-1]

                # If we are still on the same day, update the last node
                if day == current_day:
                    last_node = node_id
                else:
                    # If the day changes and it's not the first day, record the last node of the previous day
                    if current_day is not None:
                        day_nodes.append(last_node)
                    # Update the current day and reset the last node
                    current_day = day
                    last_node = node_id

            # Append the last node of the last day
            if last_node != 0:
                day_nodes.append(last_node)

            # Store the list of last nodes for each day in the result dictionary
            return day_nodes

def solve(options_df, nodes_df, time_matrix, slack, work_schedule, margin, time_limit, lunch_duration, first_optimization_algorithm, second_optimization_algorithm, GlobalSpanCostCoefficient, min_distance_overnight, start_node, verbose=False, visual=False, restrictive=False):
    #######################################################
    ######## DEFINE TIME LIMITS ###########################
    #######################################################
    if len(options_df) == 0:
        raise ValueError("No valid options found.")

    if (len(options_df) > 10) & restrictive:
        raise ValueError(f"Too many options ({len(options_df)}) - restrict prompt.")
    
    time_limit = int((time_limit - 10)/len(options_df))
    if verbose == 1:
        print('there are ', len(options_df), 'options that will be tested')
        print(f'Time limit per option: {time_limit} seconds')

    #######################################################
    ########## ITERATE OVER THE VALID OPTIONS #############
    #######################################################
    figs = {}
    route_lists_dict = {}
    options_df['respected_nodes'] = None
    options_df['fixed_appointments_nodes'] = None
    options_df['route_lists'] = None
    options_df['bad_overnights'] = None

    def main(nodes_df, time_matrix, options_df, verbose, visual, min_distance_overnight):
        for index, row in options_df.iterrows():
            if verbose:
                print("################################################")
            # blocks = row['blocks']
            blocks_cont = row['blocks_cont']
            target_keys = [1/2] + list(range(1, 8))
            new_counter = Counter()
    
            # Iterate over items in the original counter
            for key, count in blocks_cont.items():
                # Find the insertion point where this key would go in the sorted list
                pos = bisect.bisect_left(target_keys, key)
                # Get the correct key from the target keys
                new_key = target_keys[pos]
                # Add the count to the new key in the new counter
                new_counter[new_key] += count
            
            blocks_cont = new_counter
            
            gaps = row['gaps']
            gaps_cont = row['gaps_cont']
            #######################################################
            ########## CLUSTER OPTIONS ############################
            #######################################################
            clusters = custom_clustering(nodes_df, time_matrix, blocks_cont, precision=100, verbose=False, visual=False)

            if verbose == 2:
                print("########## CLUSTERING RESULTS ##########")
                for cluster, nodes in clusters.items():
                    print(f"Cluster {cluster}:")
                    print(f"Average node priority: {round(nodes_df.loc[nodes_df['node_id'].isin(nodes), 'priority'].mean(), 2)}")
                    print(f"Average distance to home: {round(nodes_df.loc[nodes_df['node_id'].isin(nodes), 'dist_to_home'].mean(), 2)}")
                    print(f"Count of nodes: {len(nodes)}")

            if visual > 1:
                plot_refined_clusters(clusters, nodes_df)

            node_to_cluster = {node: cluster for cluster, nodes in clusters.items() for node in nodes}
            nodes_df['cluster'] = nodes_df['node_id'].map(node_to_cluster)
            nodes_df['cluster_name'] = nodes_df['cluster']

            #######################################################
            ##### FIND BEST WAY TO ASSIGN NODES TO CLUSTERS #######
            ####################################################### 
            if verbose == 2:
                print("########## ASSIGNMENT ##########")
            zero_positions = [i for i, gap in enumerate(gaps) if gap == 0]
            gaps_cont = [gap for gap in gaps_cont if gap > 0]
            
            new_gaps_cont = []
    
            # Iterate over each value in the original list
            for value in gaps_cont:
                # Find the insertion point where this value would go in the sorted list
                pos = bisect.bisect_left(target_keys, value)
                # Get the correct key from the target keys
                new_key = target_keys[pos]
                # Append the new key to the new values list
                new_gaps_cont.append(new_key)
            gaps_cont = new_gaps_cont
            unique_cluster_blocks = list(nodes_df['cluster'].unique())
            solution = [[] for _ in gaps_cont]  # Initialize solution structure for each gap
            solutions = []
            if verbose:
                print("Unique cluster blocks:", unique_cluster_blocks, "Gaps cont:", gaps_cont)
            fit_blocks(unique_cluster_blocks, gaps_cont, solution, solutions, verbose=verbose)

            # Add 0s back to the solution as 'day-off' clusters
            for solution in solutions:
                for position in zero_positions:
                    solution.insert(position, ['1_day_off'])
            solutions = [[item for sublist in outer_list for item in sublist] for outer_list in solutions]
            repeated_list = []
            for sublist in solutions:
                new_sublist = []
                for item in sublist:
                    count = item.split('_')[0]
                    count = math.ceil(float(count))
                    new_sublist.extend([item] * count)  # Repeat the item
                
                repeated_list.append(new_sublist)
            
            if verbose == 2:
                print(f'Found {len(repeated_list)} solutions: {repeated_list}')

            best_count = -float('inf')
            best_list = None
            # for each list capturing the allocations of clusters to weekdays
            for i, sublist in enumerate(repeated_list):
                if verbose == 2:
                    print(f'solution {i+1}/{len(repeated_list)}')
                count = 0
                # for each cluster in the list
                for cluster in set(sublist):
                    if not 'day_off' in cluster:
                        # Access the indices of the cluster in the week
                        relevant_weekdays = [i + 1 for i, gap in enumerate(sublist) if gap == cluster]
                        
                        # get a list of non-unique fixed appointments
                        fixed_appointment_day = list(nodes_df[nodes_df['cluster'] == cluster]['weekday_fixed_appointment'])
                        
                        cluster_count_fixed = len([entry for entry in fixed_appointment_day if entry in relevant_weekdays])
                        initial_count = count
                        count += cluster_count_fixed

                        closed_days = list(nodes_df[nodes_df['cluster'] == cluster]['closed_days'].dropna())
                        flat_closed_days = []
                        for s in closed_days:
                            flat_closed_days.extend(s)
                        
                        cluster_count_closed = len([entry for entry in flat_closed_days if entry in relevant_weekdays])
                        count -= cluster_count_closed
                        if verbose == 2:
                            print(f'Relevant weekdays: {relevant_weekdays}')
                            print(f'Fixed appointment days: {fixed_appointment_day}')
                            print("Closed on:", flat_closed_days)
                            print(f'added {cluster_count_fixed} and deducted {cluster_count_closed} from {initial_count}')

                if count > best_count:
                    if verbose == 2:
                        print(f'replacing {best_count} with {count}')
                    best_count = count
                    best_list = sublist

            if verbose == 2:
                print(f'Best solution had a count of {best_count}')

            cluster_mapping_dict = {}
            for i, cluster in enumerate(best_list):
                indices = [j+1 for j, x in enumerate(best_list) if x == cluster]
                cluster_mapping_dict[cluster] = set(indices)
            nodes_df['cluster'] = nodes_df['cluster'].map(cluster_mapping_dict)
            clusters = nodes_df['cluster'].drop_duplicates().tolist()
            
            #######################################################
            ###### CHANGE ASSIGNMENT OF NODES TO CLUSTERS #########
            #######################################################
            if verbose == 2:
                print("########## UPDATE ASSIGNMENT ##########")
            # assign "to be visited on a day off" nodes to the cluster
            closed_day_problems = nodes_df[nodes_df.apply(lambda row: row['cluster'].issubset(row['closed_days']), axis=1)][['node_id', 'closed_days', 'x', 'y']]
            closed_day_problems = closed_day_problems[closed_day_problems['node_id'] != 0]
            if verbose == 2:
                    print(f'number of nodes to be reassigned: {len(closed_day_problems)}')
            for sub_row in closed_day_problems.iterrows():
                closed_on = sub_row[1]['closed_days']
                # find possible clusters (i.e.: not all closed on days contained)
                possible_clusters = [cluster for cluster in clusters if not cluster.issubset(closed_on)]
                possible_nodes = nodes_df[nodes_df['cluster'].isin(possible_clusters)]['node_id']
                possible_nodes = possible_nodes[possible_nodes != 0]
                
                if verbose == 2:
                    print(f'closed_on: {closed_on}, possible_clusters: {possible_clusters}')
                # remove 0 from row and col of time_matrix
                time_matrix_sub = time_matrix.drop(0, axis=0).drop(0, axis=1)
                closest_node = time_matrix_sub.loc[sub_row[1]['node_id'], possible_nodes].idxmin()
                closest_node_cluster = nodes_df.loc[closest_node, 'cluster']
                closest_node_cluster_name = nodes_df.loc[closest_node, 'cluster_name']

                if verbose == 2:
                    print(f'reassigning node {sub_row[1]["node_id"]} with coordinates ({sub_row[1]["x"]}, {sub_row[1]["y"]}) to cluster {closest_node_cluster} since node {closest_node} with coordinates ({nodes_df.loc[closest_node, "x"]}, {nodes_df.loc[closest_node, "y"]}) is the closest node in a possible cluster')
                # update the cluster of the node with the set defining the new cluster preventing "Must have equal len keys and value when setting with an iterable"
                nodes_df.at[sub_row[0], 'cluster'] = closest_node_cluster
                nodes_df.at[sub_row[0], 'cluster_name'] = closest_node_cluster_name
            
            # assign nodes with fixed appointments always to the cluster that contains the fixed appointments visit day in cluster index
            for index_2, row in nodes_df.iterrows():
                appointment = row['weekday_fixed_appointment']
                if (appointment not in row['cluster']) and not pd.isnull(appointment):
                    if verbose == 2:
                        print(f'{row["node_id"]} has a fixed appointment on {appointment} but is assigned to cluster {row["cluster"]}')
                    # find the cluster that contains the appointment day
                    cluster_with_appointment = [cluster for cluster in clusters if appointment in cluster][0]
                    cluster_name = nodes_df.loc[nodes_df['cluster'] == cluster_with_appointment, 'cluster_name'].iloc[0]
                    if verbose == 2:
                        print(f'Assigning node {row["node_id"]} to cluster {cluster_with_appointment}')
                    nodes_df.at[index_2, 'cluster'] = cluster_with_appointment
                    nodes_df.at[index_2, 'cluster_name'] = cluster_name

            nodes_df['adjusted_opening_hours'] = nodes_df.apply(lambda row: adjust_opening_hours(row, work_schedule, margin, time_matrix), axis=1)
            nodes_df['adjusted_opening_hours_hhmm'] = nodes_df['adjusted_opening_hours'].apply(lambda x: [[minutes_to_hhmm(item, days=True) for item in sublist] for sublist in x])
            
            # nodes_df.groupby('cluster_name') and then get list of node_ids
            clusters_and_nodes = nodes_df.groupby('cluster_name')['node_id'].apply(list)

            # add 0 to start of node_ids if it's not already in the list
            clusters_and_nodes = clusters_and_nodes.apply(lambda x: [0] + x if 0 not in x else x)
            
            #######################################################
            ########## SOLVE THE VRP FOR EACH CLUSTER #############
            #######################################################
            if verbose == 2:
                print("########## SOLVE VRP ##########")
            dropped_nodes = []
            route_lists = {}
            obj_values = 0
            for cluster, node_ids in clusters_and_nodes.items():
                sub_nodes_df = nodes_df[nodes_df['node_id'].isin(node_ids)]
                print(sub_nodes_df['cluster'])
                cluster_set = sub_nodes_df['cluster'].values[1]
                if verbose == 3:
                    print(f'##### {cluster} #####')
                    print(f'initial nodes: {node_ids}')
                if dropped_nodes:
                    # dropped nodes that were fixed appointments
                    dropped_nodes_df = nodes_df[(nodes_df['node_id'].isin(dropped_nodes)) & (nodes_df['fixed_appointment'].notnull())]
                    if verbose == 3:
                        print(f'dropped nodes: {dropped_nodes} of which with fixed appointments: {dropped_nodes_df["node_id"].tolist()}')
                    dropped_nodes_df = nodes_df[nodes_df['node_id'].isin(dropped_nodes)]
                    sub_nodes_df = pd.concat([sub_nodes_df, dropped_nodes_df])
                
                sub_nodes_df.loc[:, 'cluster_name'] = cluster
                sub_nodes_df.loc[:, 'cluster'] = sub_nodes_df.loc[:, 'cluster'].apply(lambda _: cluster_set)
                sub_nodes_df['adjusted_opening_hours'] = sub_nodes_df.apply(lambda row: adjust_opening_hours(row, work_schedule, margin, time_matrix), axis=1)
                sub_nodes_df['adjusted_opening_hours_hhmm'] = sub_nodes_df['adjusted_opening_hours'].apply(lambda x: [[minutes_to_hhmm(item, days=True) for item in sublist] for sublist in x])



                # Since nodes are assigned to clusters visiting on days on which they are not open (adjusted_opening_hours is empty list), we have to add those to dropped nodes for a potnetial next iteration and remove them from the current sub_nodes_df
                invalid_rows = sub_nodes_df[sub_nodes_df['adjusted_opening_hours'].apply(lambda x: len(x) == 0)]
                if not invalid_rows.empty:
                    if verbose:
                        print(f"Nodes {invalid_rows['node_id'].tolist()} are assigned to a cluster on a day they are not open.")
                    dropped_nodes.extend(invalid_rows['node_id'].tolist())
                    dropped_nodes = list(set(dropped_nodes))
                    sub_nodes_df = sub_nodes_df[~sub_nodes_df.index.isin(invalid_rows.index)]


                # remove and warn about rows that have a weekday_fixed_appointment that is not in the cluster
                invalid_rows = sub_nodes_df[sub_nodes_df.apply(
                    lambda row: not pd.isna(row['weekday_fixed_appointment']) and row['weekday_fixed_appointment'] not in row['cluster'], axis=1)]
                if not invalid_rows.empty:
                    if verbose > 1:
                        print(f"Nodes {invalid_rows['node_id'].tolist()} have fixed appointments that can not take place on the assigned days.")
                        print(f'PROOF: {invalid_rows[['weekday_fixed_appointment', 'cluster']]}')
                    sub_nodes_df = sub_nodes_df[~sub_nodes_df.index.isin(invalid_rows.index)]
                # find route -> dropped nodes
                if verbose == 3:
                    print(f'nodes for finding route: {sub_nodes_df["node_id"].tolist()}')
                # make time limit per cluster dependent on the fraction of fixed appointments consdidered
                total_fixed = sum(nodes_df['fixed_appointment'].notnull())
                fraction = len(sub_nodes_df) / len(nodes_df)
                if total_fixed == 0:
                    time_limit_cluster = int(time_limit * fraction)
                else:
                    fraction_fixed_appointments = sum(sub_nodes_df['fixed_appointment'].notnull()) / total_fixed
                    time_limit_cluster = int(time_limit * fraction_fixed_appointments)
                if time_limit_cluster < 2:    
                    time_limit_cluster = 2
                if verbose == 1:
                    print(f'spending {time_limit_cluster} seconds on cluster {cluster} of option {index} since it has {sum(sub_nodes_df["fixed_appointment"].notnull())} fixed appointments out of {sum(nodes_df["fixed_appointment"].notnull())}')
                
                result, dropped_nodes_iteration, obj_value = solve_vrp(time_matrix, sub_nodes_df, slack, lunch_duration, first_optimization_algorithm, second_optimization_algorithm, GlobalSpanCostCoefficient, work_schedule, time_limit_cluster, start_node, verbose=True)
                
                if dropped_nodes_iteration:
                    if dropped_nodes:
                        dropped_nodes.extend(dropped_nodes_iteration)
                    else:
                        dropped_nodes = dropped_nodes_iteration
                    dropped_nodes = list(set(dropped_nodes))
                if not invalid_rows.empty:
                    dropped_nodes.extend(invalid_rows['node_id'].tolist())
                    dropped_nodes = list(set(dropped_nodes))
                obj_values += obj_value
                
                route_lists[cluster] = result[1] # route without travel
                if verbose == 1:
                    print(f'Route for {cluster}: {result[0]}')
            
            #######################################################
            ########## EVALUATE THE RESULTS #######################
            #######################################################            
            updated_routes = {}
            for trip_key in route_lists:
                days = str(cluster_mapping_dict[trip_key])
                updated_key = f"{trip_key}_{days}"
                updated_routes[updated_key] = route_lists[trip_key]
            route_lists_dict[index] = updated_routes
            overnight_nodes = find_last_nodes_per_day(updated_routes)
            fig = plot_all_cluster_routes(updated_routes, nodes_df, overnight_nodes)
            bad_overnight = []
            if overnight_nodes:
                for node in overnight_nodes:
                    distance = time_matrix.iloc[0, node]
                    if distance < min_distance_overnight:
                        bad_overnight.append(node)
            if len(bad_overnight) > 0:
                print("WARNING: Overnight nodes are too close to the home node.")
            
            figs[index] = fig
            if visual:
                print(f'Option {index} with objective value {obj_values} and {len(dropped_nodes)} dropped nodes')
                fig.show()

            all_nodes = set(nodes_df['node_id'])
            respected_nodes_route = set()
            for trip in route_lists:
                    route = route_lists[trip]
                    for node in route:
                        respected_nodes_route.add(node[0])
            all_dropped_nodes = all_nodes - respected_nodes_route
            fixed_appointment_nodes = set(nodes_df[nodes_df['fixed_appointment'].notnull()]['node_id'])

            options_df.at[index, 'respected_nodes'] = respected_nodes_route
            options_df.at[index, 'nodes_cons'] = 1 - (len(all_dropped_nodes) / len(all_nodes))
            options_df.at[index, 'num_nodes_cons'] = len(all_nodes) - len(all_dropped_nodes)
            if len(fixed_appointment_nodes) == 0:
                options_df.at[index, 'fixed_app_cons'] = 1
            else:
                options_df.at[index, 'fixed_app_cons'] = len(respected_nodes_route & fixed_appointment_nodes) / len(fixed_appointment_nodes)
            
            options_df.at[index, 'obj_value'] = obj_values

            options_df.at[index, 'num_nodes_considered'] = len(options_df.at[index, 'respected_nodes'])
            
            options_df.at[index, 'route_lists'] = route_lists
            options_df.at[index, 'bad_overnights'] = bad_overnight
            
            options_df.at[index, 'fixed_appointments_nodes'] = list(fixed_appointment_nodes)
            options_df.at[index, 'fixed_appointments_respected'] = f"{len(fixed_appointment_nodes) - len(set(dropped_nodes) & set(fixed_appointment_nodes))} out of {len(fixed_appointment_nodes)}"

        return options_df, figs, route_lists_dict
    
    options_df, figs, route_lists_dict = main(nodes_df, time_matrix, options_df, verbose, visual, min_distance_overnight)

    options_df = options_df[['obj_value', 'n_overnight_trips', 'nodes_cons', 'fixed_app_cons', 'num_nodes_cons', 'bad_overnights', 'route_lists', 'gaps', 'Sum', 'Length', 'num_nodes_considered', 'overnight_days', 'off_days', 'n_days_off', 'blocks', 'respected_nodes', 'fixed_appointments_nodes']]
    
    return options_df, figs, route_lists_dict, nodes_df

def prelim_check(nodes_df, time_matrix, work_schedule, fix_app_margin, work_days, verbose=False):
    messages = []

    # ONE DAY WEEK OR DYNAMIC CALCULATION PROBLEMS WITH CLOSED DAYS
    if len(work_days) == 1:
        bad_ids = nodes_df[nodes_df['open_days'].apply(lambda x: len(set(x).intersection(work_days)) == 0)]['node_id'].values.tolist()
        bad_ids = [x for x in bad_ids if x != 0]
        nodes_df = nodes_df[~nodes_df['node_id'].isin(bad_ids)]
        messages.append(f"Nodes closed on days they are assigned to (removed due to one day week): {bad_ids}")

    # FIXED APPOINTMENT CONFLICTS
    fixed_appointments = nodes_df[nodes_df['fixed_appointment'].notnull()]

    appointments = [
        (row['node_id'], row['fixed_appointment'][0], row['fixed_appointment'][1], row['fixed_appointment'][2])
        for _, row in fixed_appointments.iterrows()
    ]

    def parse_time(time_input):
        """Converts a time input (string or datetime.time) to minutes since midnight."""
        if isinstance(time_input, str):
            t = datetime.strptime(time_input, '%H:%M:%S').time()
        elif isinstance(time_input, time):
            t = time_input
        else:
            raise ValueError("Invalid time input. Must be a string or datetime.time object.")
        return t.hour * 60 + t.minute + t.second // 60

    def find_out_of_schedule_appointments(appointments, work_schedule):
        """
        Finds nodes with appointments that either:
        1. Don't fit within the specified work hours, or
        2. Can't be reached on time from the home node or can't return to home node before the end of work hours.
        """
        out_of_schedule_nodes = []

        for node_id, weekday, start_str, end_str in appointments:
            start = parse_time(start_str) - fix_app_margin
            end = parse_time(end_str)
            work_start, work_end = work_schedule.get(weekday, [0, 0])

            # Calculate the travel time to and from the home node
            travel_time = time_matrix.iloc[0, node_id]

            # Convert travel times to start and end of the appointment in minutes since midnight
            actual_start = start - travel_time
            actual_end = end + travel_time

            # Check if the appointment start or end is outside work hours including travel time
            if actual_start < work_start or actual_end > work_end:
                out_of_schedule_nodes.append(node_id)

        return out_of_schedule_nodes

    def find_conflicts(appointments):
        # Sort appointments by weekday and start time (converted to minutes)
        appointments.sort(key=lambda x: (x[1], parse_time(x[2])))
        conflicts = []

        for i in range(len(appointments)):
            for j in range(i + 1, len(appointments)):
                id1, day1, start1, end1 = appointments[i]
                id2, day2, start2, end2 = appointments[j]

                # Fetch on_site_time for the first node
                on_site_time = nodes_df.loc[nodes_df['node_id'] == id1, 'on_site_time'].values[0]

                # Only consider appointments on the same day
                if day1 != day2:
                    continue

                # Convert times to minutes since midnight for comparison
                start1 = parse_time(start1) - fix_app_margin
                end1 = parse_time(end1) + on_site_time  # Include on-site time in the end time
                start2 = parse_time(start2) - fix_app_margin
                end2 = parse_time(end2)

                # Check for overlapping times
                if start1 < end2 and start2 < end1:
                    conflicts.append([id1, id2])
                    continue
                
                # Check if there's enough time between appointments considering travel time
                travel_time = timedelta(minutes=time_matrix.iloc[id1, id2])
                travel_time_minutes = travel_time.total_seconds() / 60

                # Calculate time difference from the end of the first appointment to the start of the second
                time_diff = start2 - end1

                if time_diff < travel_time_minutes:
                    conflicts.append([id1, id2])

        return conflicts


    def conflicting_nodes_to_be_removed(conflicts):
        node_counter = Counter(node for conflict in conflicts for node in conflict)
        removed_nodes = []
        while conflicts:
            max_node = max(node_counter, key=node_counter.get)
            removed_nodes.append(max_node)
            conflicts = [conflict for conflict in conflicts if max_node not in conflict]
            node_counter = Counter(node for conflict in conflicts for node in conflict)
        return removed_nodes

    def check_appointment_fit(appointment, opening_hours):
        """Checks if the appointment fits within any of the opening intervals."""
        start, end = parse_time(appointment[1]) - fix_app_margin, parse_time(appointment[2])
        day_opening_hours = opening_hours.get(appointment[0], [])
        
        for interval in day_opening_hours:
            interval_start, interval_end = parse_time(interval[0]), parse_time(interval[1])
            if start >= interval_start and end <= interval_end:
                return True
        return False

    def find_conflicting_appointments(appointments, opening_hours):
        """Finds nodes with appointments that don't fit in the opening hours."""
        conflicting_nodes = []
        
        for node_id, weekday, start_time, end_time in appointments:
            node_opening_hours = opening_hours.get(node_id, {})
            
            if not check_appointment_fit((weekday, start_time, end_time), node_opening_hours):
                conflicting_nodes.append(node_id)

        return conflicting_nodes

    # Conflicting fixed appointments WORK HOURS
    out_of_schedule_nodes = find_out_of_schedule_appointments(appointments, work_schedule)
    messages.append(f"Appointments not within the work schedule: {out_of_schedule_nodes}")
    nodes_df = nodes_df[~nodes_df['node_id'].isin(out_of_schedule_nodes)]
    appointments = [appointment for appointment in appointments if appointment[0] not in out_of_schedule_nodes]

    # Conflicting fixed appointments OPENING HOURS
    opening_hours = nodes_df.set_index('node_id')['opening_hours'].to_dict()
    conflicting_nodes = find_conflicting_appointments(appointments, opening_hours)
    messages.append(f"Appointments not within opening hours of node: {conflicting_nodes}")
    nodes_df = nodes_df[~nodes_df['node_id'].isin(conflicting_nodes)]
    appointments = [appointment for appointment in appointments if appointment[0] not in conflicting_nodes]

    # Conflicting fixed appointments AMONG
    conflicts = find_conflicts(appointments)
    nodes_to_be_removed = conflicting_nodes_to_be_removed(conflicts)
    messages.append(f'Appointments removed due to conflicts among fixed appointments: {nodes_to_be_removed}')
    nodes_df = nodes_df[~nodes_df['node_id'].isin(nodes_to_be_removed)]

    if verbose:
        print("\n".join(messages))
    return nodes_df, messages
import streamlit as st
from datetime import datetime

import streamlit as st
from datetime import datetime, time

from src.routing import solve, create_data

st.set_page_config(layout="wide")

# Helper function to calculate the percentage of a normal workday
def calculate_day_percentage(start, end):
    if start and end:
        total_hours = (datetime.combine(datetime.today(), end) - datetime.combine(datetime.today(), start)).seconds / 3600
        return (total_hours / 8.5) * 100
    return 0

# App title and header
st.title("Route Planner Dashboard")
st.header("Input")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Settings")
    num_nodes = st.number_input("Enter Number of Nodes", min_value=1, value=50, step=1)
    percentage_of_appointments = st.slider("Percentage of fixed appointments", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    min_work_days = st.number_input("Minimum number of work days per location", min_value=1, max_value=7, value=7)
    penalty_factor = st.number_input("Penalty", min_value=0.0, max_value=100000000.0, value=10000.0, step=0.1)

# if st.session_state['nodes_df'] and st.session_state['time_matrix'] don't exist, create them
if 'nodes_df' not in st.session_state or 'time_matrix' not in st.session_state:
    nodes_df, time_matrix = create_data(min_work_days, {}, percentage_of_appointments, num_nodes)
    st.session_state['nodes_df'] = nodes_df
    st.session_state['time_matrix'] = time_matrix

with col2:
    if st.button('Recreate data'):
        if st.session_state['days_off']:
            days_off = st.session_state['days_off']
        else:
            days_off = {}
        nodes_df, time_matrix = create_data(min_work_days, days_off, percentage_of_appointments, num_nodes, penalty_factor)
        
        st.session_state['nodes_df'] = nodes_df
        st.session_state['time_matrix'] = time_matrix
    if len(st.session_state['nodes_df']) > 0:
        nodes_df = st.session_state['nodes_df']
        st.dataframe(nodes_df)

st.header("Work Preferences")
max_days_off = st.number_input("Maximum days off allowed", min_value=0, max_value=7, value=0)
max_short_days = st.number_input("Maximum short days allowed", min_value=0, max_value=7, value=1)
max_overnight_stays = st.number_input("Maximum overnight stays allowed", min_value=0, max_value=7, value=2)
lunch_duration = st.number_input("Default lunch duration (minutes)", min_value=0, max_value=120, value=30)

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

with st.container():
    cols = st.columns([1, 1, 1, 1, 1, 1, 1])  # Adjust column widths as necessary

    # Create headers for the table
    headers = ['Day', 'Work START', 'Work END', 'Overnight OK', 'Day FREE', '% of Normal Day']
    for i, header in enumerate(headers):
        cols[i].write(f"**{header}**")

    # Create input fields for each day
    week_config = []
    for day in weekdays:
        # Use st.container() to align inputs
        with st.container():
            row_cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 2])
            row_cols[0].write(f"**{day}**")
            start_time = row_cols[1].time_input("Start Time", key=f"start_{day}", value=time(8, 0), label_visibility='hidden')
            end_time = row_cols[2].time_input("End Time", key=f"end_{day}", value=time(18, 0), label_visibility='hidden')
            overnight_ok = row_cols[3].checkbox("placeholder_label", key=f"overnight_ok_{day}", label_visibility='hidden')
            # no_overnight = row_cols[4].checkbox("", key=f"no_overnight_{day}")
            day_free = row_cols[5].checkbox("placeholder_label", key=f"day_free_{day}", label_visibility='hidden')
            day_percentage = calculate_day_percentage(start_time, end_time)
            row_cols[7].write(f"{day_percentage:.2f}%")

            week_config.append({
                "Day": day,
                "Work START": start_time,
                "Work END": end_time,
                "Overnight OK": overnight_ok,
                # "NO Overnight": no_overnight,
                "Day FREE": day_free,
                "% of Normal Day": f"{day_percentage:.2f}%"
            })
    # week_df = pd.DataFrame(week_config)
    # st.dataframe(week_df)

# Transform / derive inputs
days_off = {i + 1 for i, day in enumerate(week_config) if day["Day FREE"]}
st.session_state['days_off'] = days_off
short_days_threshold = 5  # hours
short_days = {i + 1 for i, day in enumerate(week_config) if (day["Work END"].hour - day["Work START"].hour) < short_days_threshold}
no_overnight_stays = {i + 1 for i, day in enumerate(week_config) if not day["Overnight OK"]}
work_schedule = {int(i + 1): [day["Work START"].hour * 60 + day["Work START"].minute, day["Work END"].hour * 60 + day["Work END"].minute] for i, day in enumerate(week_config)}

# Developer settings
st.subheader("Developer Settings")
col1, col2 = st.columns(2)

with col1:
    optimization_algorithm = st.selectbox("Optimization algorithm", ["PATH_CHEAPEST_ARC", "PARALLEL_CHEAPEST_INSERTION", "SAVINGS", 'BEST_INSERTION'], index=1)
    slack = st.number_input("Slack (max. idle time)", value=100)

with col2:
    GlobalSpanCostCoefficient = st.number_input("Global Span Cost Coefficient", min_value=0, max_value=1000000, value=500, step=1)
    margin = st.number_input("Margin (arrival before appointment)", value=5)

st.header("Possible Routes")
if st.button('Run'):
    time1 = datetime.now()
    if len(no_overnight_stays) > 0:
        possible_overnight_stays = set(range(1, 8)) - no_overnight_stays
    else:
        possible_overnight_stays = set(range(1, 8))
    max_short_days = 0
    short_days = {day for day, hours in work_schedule.items() if (hours[1] - hours[0]) / 60 <= 4}

    messages = []
    removed_days = [day for day in list(days_off) if day in short_days]
    for day in removed_days:
        days_off.remove(day)
        messages.append(f"Removed day {day} from days off because it's a short day or overnight stay.")

    # Check for more days off defined than max days off
    if len(days_off) > max_days_off:
        max_days_off = len(days_off)
        messages.append(f"Adjusted max days off to {max_days_off} because {len(days_off)} days off were defined.")

    # Check for more short days defined than max short days
    if len(short_days) > max_short_days:
        max_short_days = len(short_days)
        messages.append(f"Adjusted max short days to {max_short_days} because {len(short_days)} short days were defined.")

    # if any fixed appointments on days off, remove them
    nodes_df = st.session_state['nodes_df']
    messages.append(f"Removed {sum(nodes_df['weekday_fixed_appointment'].isin(days_off))} fixed appointments on days off.")
    nodes_df = nodes_df[~nodes_df['weekday_fixed_appointment'].isin(days_off)]
    
    routes_df, figs = solve(nodes_df, st.session_state['time_matrix'], slack, penalty_factor, days_off, max_days_off, short_days, max_short_days, no_overnight_stays, max_overnight_stays, work_schedule, margin, lunch_duration, optimization_algorithm, GlobalSpanCostCoefficient, verbose=False, visual=False, restrictive=True, deep=False)

    time2 = datetime.now()
    time_taken = time2 - time1
    time_taken = round(time_taken.total_seconds(), 2)
    if len(routes_df) == 1:
        route_word = "opition"
    else:
        route_word = "opitions"
    st.write(f"Tested {len(routes_df)} {route_word} in {time_taken} Seconds.")
    if len(messages) > 0:
        st.write(f"{messages}")
    st.session_state['routes_df'] = routes_df
    st.session_state['figs'] = figs

if 'routes_df' in st.session_state:
    routes_df = st.session_state['routes_df']
    figs = st.session_state['figs']

    if len(routes_df) > 0:
        st.dataframe(routes_df)
        selected_index = st.selectbox("Select route to show:", routes_df.index)
        st.header("Route Visualization")
        st.plotly_chart(figs[selected_index], use_container_width=True)
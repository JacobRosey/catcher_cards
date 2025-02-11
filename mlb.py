import pybaseball
from pybaseball import statcast_catcher_framing, statcast_catcher_poptime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import requests
from PIL import Image
import requests
from io import BytesIO
import tkinter as tk
import statsapi
import os
import ast

# Function to get player index from framing data
def get_player_index(user_in, unfiltered_framing):
    for index, row in unfiltered_framing.iterrows():
        if index == 0:  # Skip league average row
            continue
        print(f"index: {index} player: {row.first_name} {row.last_name}")
        if user_in.lower() == (row.first_name + ' ' + row.last_name).lower():
            return index
    return None

# Function to get player index from poptime data
def get_pop_player_index(player_name, unfiltered_pop):
    formatted_name = f"{player_name.split()[1]}, {player_name.split()[0]}"
    for index, row in unfiltered_pop.iterrows():
        if formatted_name.lower() == row.catcher.lower():
            return index
    return None

# Function to calculate percentile mappings
def get_percentile_mapping(column):

    filtered_cols = ['id', 'year', 'catcher', 'last_name', 'first_name', 'player_id', 'team_id', 'age']
    if column.name in filtered_cols:
        return None # we don't give no fucks about your age percentile
    percentiles = np.arange(0, 101, 5)  # 0th to 100th percentiles, steps of 5
    values = np.percentile(column.dropna(), percentiles)
    return dict(zip(percentiles, values))

# Function to find the percentile of a value
def find_percentile(value, mapping):
    for percentile, threshold in sorted(mapping.items()):
        if value <= threshold:
            return percentile
    return 100

def get_player_headshot(player_id: str):
    # Construct the URL for the player's headshot image
    url = f'https://img.mlbstatic.com/mlb-photos/image/'\
          f'upload/d_people:generic:headshot:67:current.png'\
          f'/w_640,q_auto:best/v1/people/{player_id}/headshot/silo/current.png'

    # Send a GET request to the URL
    response = requests.get(url)

    # Open the image from the response content
    img = Image.open(BytesIO(response.content))

    return img

def get_catcher_throwing(player_id, year):
    url = f'https://baseballsavant.mlb.com/leaderboard/services/catcher-throwing/{player_id}?game_type=Regular&n=q&season_end={year}&season_start={year}&split=no&team=&type=Cat&with_team_only=1'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the response is successful
    if response.status_code == 200:
        # Parse JSON content from the response
        return pd.DataFrame(response.json())  # This directly returns the parsed JSON as a Python object
    else:
        print(f"Error: {response.status_code}")
        return None

def create_strike_zone_plot(ax, catcher_framing, get_grid_color):
    # Draw the strike zone grid
    ax.add_patch(patches.Rectangle((0, 0), 3, 3, edgecolor='black', facecolor='none'))
    ax.add_patch(patches.Rectangle((1, 1), 1, 1, edgecolor='black', facecolor='none'))

    ax.add_patch(patches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_17')['color']))
    ax.add_patch(patches.Rectangle((2, 0), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_19')['color']))
    ax.add_patch(patches.Rectangle((0, 2), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_11')['color']))
    ax.add_patch(patches.Rectangle((2, 2), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_13')['color']))

    ax.add_patch(patches.Rectangle((1, 0), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_18')['color']))
    ax.add_patch(patches.Rectangle((1, 2), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_12')['color']))
    ax.add_patch(patches.Rectangle((0, 1), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_14')['color']))
    ax.add_patch(patches.Rectangle((2, 1), 1, 1, edgecolor='black', facecolor=get_grid_color('strike_rate_16')['color']))

    # Set axis limits and labels
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes

    # Add labels
    ax.text(0.5, 2.5, f"{catcher_framing.strike_rate_11}%", ha='center', va='center', fontsize=10)
    ax.text(0.5, 2.2,  f"{get_grid_color('strike_rate_11')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(1.5, 2.5, f"{catcher_framing.strike_rate_12}%", ha='center', va='center', fontsize=10)
    ax.text(1.5, 2.2, f"{get_grid_color('strike_rate_12')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(2.5, 2.5, f"{catcher_framing.strike_rate_13}%", ha='center', va='center', fontsize=10)
    ax.text(2.5, 2.2, f"{get_grid_color('strike_rate_13')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(0.5, 1.5, f"{catcher_framing.strike_rate_14}%", ha='center', va='center', fontsize=10)
    ax.text(0.5, 1.2, f"{get_grid_color('strike_rate_14')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(1.5, 1.5, f"FR: {catcher_framing.runs_extra_strikes}", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 1.2, f"{get_grid_color('runs_extra_strikes')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(2.5, 1.5, f"{catcher_framing.strike_rate_16}%", ha='center', va='center', fontsize=10)
    ax.text(2.5, 1.2, f"{get_grid_color('strike_rate_16')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(0.5, 0.5, f"{catcher_framing.strike_rate_17}%", ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.2, f"{get_grid_color('strike_rate_17')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(1.5, 0.5, f"{catcher_framing.strike_rate_18}%", ha='center', va='center', fontsize=10)
    ax.text(1.5, 0.2, f"{get_grid_color('strike_rate_18')['value']}th", ha='center', va='center', fontsize=8)
    ax.text(2.5, 0.5, f"{catcher_framing.strike_rate_19}%", ha='center', va='center', fontsize=10)
    ax.text(2.5, 0.2, f"{get_grid_color('strike_rate_19')['value']}th", ha='center', va='center', fontsize=8)

def create_csaa_plot(ax, csaa, csaa_percentile_mappings):
    #categories = ['CSAA', 'CSAA-teamwork', 'Pure CSAA']
    #values = csaa[0], csaa[1], csaa[2]
    categories = ['Pure CSAA', 'CSAA']
    values = csaa[2], csaa[0]
    
    # Calculate the percentiles for each category
    percentiles = [
        find_percentile(csaa[2], csaa_percentile_mappings['csaa']),
        #find_percentile(csaa[1], csaa_percentile_mappings['csaa-team']),
        find_percentile(csaa[0], csaa_percentile_mappings['pure_csaa'])
    ]
    
    # Define the color based on the percentile
    def get_color(percentile):
        if percentile < 40:
            return 'crimson'  # Red
        elif percentile < 55:
            return 'yellow'  # Yellow
        elif percentile < 70:
            return 'lightgreen'  # Light Green
        else:
            return 'green'  # Dark Green
    
    # Assign a color for each bar based on the percentile
    colors = [get_color(p) for p in percentiles]
    
    # Create horizontal bar chart with dynamic colors
    bars = ax.barh(categories, values, color=colors, height=0.5)  # Set height to create space between bars
    
    # Set axis limits and remove axis
    ax.set_xlim(-10, 10)  # Center the axis at 0 and allow both negative and positive values
    ax.axis('off')
    
   # Add text labels to each bar
    for bar, value, percentile in zip(bars, values, percentiles):
        # For negative values, place the value label to the left, and for positive values to the right
        if value < 0:
            # Ensure the label does not go too far left (beyond -10)
            label_x = bar.get_width() - 0.4
            if bar.get_width() < -9.6:  # Check if the bar is close to the left edge (-10)
                label_x = -10.4  # Place label outside the bar
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,  # Position text to the left for negative
                    f'{value:.2f}', va='center', fontsize=10, ha='right')
            # Add the percentile near the origin of each bar
            ax.text(4.5, bar.get_y() + bar.get_height() / 2,  # Position percentile at the center (origin)
                f'{percentile}th percentile', va='center', ha='center', fontsize=10)
        else:
            # Ensure the label does not go too far right (beyond 10)
            label_x = bar.get_width() + 0.4
            if bar.get_width() > 9.6:  # Check if the bar is close to the right edge (10)
                label_x = 10.4  # Place label outside the bar
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,  # Position text to the right for positive
                    f'{value:.2f}', va='center', fontsize=10, ha='left')
            # Add the percentile near the origin of each bar
            ax.text(-4.5, bar.get_y() + bar.get_height() / 2,  # Position percentile at the center (origin)
                f'{percentile}th percentile', va='center', ha='center', fontsize=10)

    # Move category labels above the bars, closer to the origin
    for index, category in enumerate(categories):
        
        # Position category label above the bar, centered over each bar
        ax.text(0, bars[index].get_y() + bars[index].get_height() / 2 + 0.4,  # Slightly above the center of the bar
                category, va='center', ha='center', fontsize=10, fontweight='bold')


def create_bar_plot(ax):
    categories = ['a', 'b', 'c', 'd']
    values = [1, 2, 3, 4]
    ax.bar(categories, values, color='crimson')
    ax.set_ylim(0, 10)
    ax.axis('off')

def show_logo(ax): #figure out how to round the corners of the image
    img = mpimg.imread('kickdirt.jpg')  
    ax.imshow(img, aspect='auto', extent=[-0.125, 0.125, 0.75, 1.0], zorder=10)  # Adjust `extent` for placement
    ax.axis('off')

# replace args with player_info object with image, name, age, height, weight
def create_headshot_info_plot(ax, player_name, img):

    ax.imshow(img, aspect='auto', extent=[0.2, 0.3, 0.4, 0.1])  # Adjust extent to position the image
    ax.text(0.5, 1.0, f"{player_name}", ha='center', va='center', fontsize=12)
    #ax.text(0.25, 0.2, "Age", ha='center', va='center', fontsize=10)
    #ax.text(0.5, 0.2, "height", ha='center', va='center', fontsize=10)
    #ax.text(0.75, 0.2, "weight", ha='center', va='center', fontsize=10)
    ax.axis('off')  # Hide axes

def create_key_stats_plot(ax, stats23, stats24):
        
    cell_text = [
        ["", "2023", "2024"],
        ["Innings", stats23.get("innings", "No data"), stats24.get("innings", "No data")],
        ["Fielding %", stats23.get("fielding", "No data"), stats24.get("fielding", "No data")],
        ["E", stats23.get("errors", "No data"), stats24.get("errors", "No data")],
        #["WP", stats23.get("wildPitches", "No data"), stats24.get("wildPitches", "No data")],
        ["PB", stats23.get("passedBall", "No data"), stats24.get("passedBall", "No data")],
        ["DP", stats23.get("doublePlays", "No data"), stats24.get("doublePlays", "No data")],
        #["CERA", stats23.get("catcherERA", "No data"), stats24.get("catcherERA", "No data")]
    ]

    # Continue with the code for displaying the table
    table = ax.table(cellText=cell_text, loc='center', cellLoc='center')

    for key, cell in table.get_celld().items():
        cell.set_height(.18)  # Adjust cell height
        cell.set_width(.32)   # Adjust cell width

    # Highlight the header row
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == 0:  # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#f0f0f0')  # Light gray background for header
           
    # Add a placeholder for the image in the top-left cell
    ax.text(0.15, 0.85, "", ha='center', va='center', fontsize=8, color='black', transform=ax.transAxes)

    # Remove axes for better visibility
    ax.axis('off')

def get_catcher_stats(player_data, year):
    # Find the stats for 2024 where the position is "Catcher"
    for stat_entry in player_data['stats']:
        if stat_entry['season'] == year and stat_entry['stats']['position']['name'] == 'Catcher':
            return stat_entry['stats']
    return None

def get_current_team(player_data):
    return player_data['current_team'] or None

def collect_player_ids(catchers):
    ids = []
    for index, row in catchers.iterrows():
        if(index == 0):continue
        ids.append(int(row.player_id))
    return ids


def get_all_catcher_throwing(ids, filename='catcher_throwing_data.csv'):
    # Check if the file already exists
    if os.path.exists(filename):
        # If the file exists, just read and return its contents
        print(f"File {filename} exists. Returning the data from the file.")
        return pd.read_csv(filename)

    # If the file does not exist, fetch the data and save it
    catcher_throwing_list = []
    index = 0
    for id in ids:
        print(f"Fetching catcher stats for {index} player_id: {id}")
        catcher_throwing_data = get_catcher_throwing(id, '2024')

        # If the returned data is a dictionary, convert it to a DataFrame
        if isinstance(catcher_throwing_data, dict):
            # Flatten the dictionary before appending
            catcher_throwing_data = pd.json_normalize(catcher_throwing_data)

        catcher_throwing_list.append(catcher_throwing_data)
        index += 1

    # Concatenate all DataFrames at once
    all_catcher_throwing = pd.concat(catcher_throwing_list, ignore_index=True)

    # Save the data to a CSV file
    all_catcher_throwing.to_csv(filename, index=False, header=True)
    
    return all_catcher_throwing  # Return the flattened DataFrame


def get_csaa(catcher_throwing):
    csaa = 0
    teamwork = 0
    pitching = 0

    for _, row in catcher_throwing.iterrows():
        data = row.get('data', [])  # Assuming 'data' is a column with nested data
        csaa += data.get('cs_aa', 0)
        teamwork += data.get('teamwork_over_xcs', 0)
        if type(data.get('pitcher_cs_aa')) == float:
            pitching += data.get('pitcher_cs_aa', 0)
        
    return [csaa, csaa - teamwork, csaa - pitching - teamwork]

#refactor to work with a loop to get_csaa
def get_all_csaa(all_catcher_throwing):
    all_csaa = []

    current_player_id = current_csaa = current_teamwork = current_pitching = 0

    # convert string to dict
    all_catcher_throwing['data'] = all_catcher_throwing['data'].apply(ast.literal_eval)

    for index, catcher in all_catcher_throwing.iterrows():
        data = catcher.get('data')
        this_player_id = data.get('catcher_id')
        if current_player_id != this_player_id:
            print(f'last player id: {current_player_id} next player id: {this_player_id}')
            all_csaa.append([current_player_id, current_csaa, current_csaa - current_teamwork, current_csaa - current_teamwork - current_pitching  ])
            current_player_id = this_player_id
            current_csaa = current_teamwork = current_pitching = 0

        current_csaa += data.get('cs_aa')
        current_teamwork += data.get('teamwork_over_xcs')
        current_pitching += data.get('pitcher_cs_aa') if type(data.get('pitcher_cs_aa')) == float else 0
    all_csaa.append([current_player_id, current_csaa, current_csaa - current_teamwork, current_csaa - current_teamwork - current_pitching  ])
    return pd.DataFrame(all_csaa, columns=['player_id', 'csaa', 'csaa-team', 'pure_csaa'])    

def create_csaa_scatter_plot(all_csaa):
    # Calculate the difference between csaa and pure_csaa for the x-axis
    all_csaa['csaa_difference'] = round(all_csaa['csaa'] - all_csaa['pure_csaa'])
    
    # Filter out players whose CSAA is too close to the origin (near 0), except for Austin Wells
    all_csaa_filtered = all_csaa[
    ~((all_csaa['csaa'].abs() <= 4) | (all_csaa['csaa_difference'].abs() < 2) | 
      (all_csaa['last_name'] == "Knizner") | (all_csaa['last_name'] == "Ruiz"))
]

    # Specific additions/removals by request
    aw_row = all_csaa[(all_csaa['first_name'] == 'Austin') & (all_csaa['last_name'] == 'Wells')]
    wc_row = all_csaa[(all_csaa['first_name'] == 'William') & (all_csaa['last_name'] == 'Contreras')]
    yd_row = all_csaa[(all_csaa['first_name'] == 'Yainer') & (all_csaa['last_name'] == 'Diaz')]
    fa_row = all_csaa[(all_csaa['first_name'] == 'Francisco') & (all_csaa['last_name'] == 'Alvarez')]
    all_csaa_filtered = pd.concat([all_csaa_filtered, aw_row, wc_row, yd_row, fa_row])


    pd.set_option('display.max_rows', None)  # No limit on the number of rows
    pd.set_option('display.max_columns', None)  # No limit on the number of columns
    pd.set_option('display.width', None)  # Disable line wrapping for wide dataframes 
    pd.set_option('display.max_colwidth', None)  # No truncation of column content
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot: x is the difference between CSAA and Pure CSAA, y is the CSAA value
    scatter = ax.scatter(all_csaa_filtered['csaa_difference'], all_csaa_filtered['csaa'], color='blue', alpha=0.5, edgecolors='w', s=100)
    
    # Add labels and title
    ax.set_xlabel('Teamwork Impact', fontsize=12)
    ax.set_ylabel('CSAA', fontsize=12)
    ax.set_title("Teamwork's Impact on CSAA", fontsize=14)
    
    # Add a grid for better visualization
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-12, 14)
    
    # Loop through each player in the filtered dataframe to add their name to the plot
    for i, row in all_csaa_filtered.iterrows():
        # Add player's name above their point (adjust y-offset to avoid overlap)
        ax.text(row['csaa_difference'], row['csaa'] + 0.3, f"{row['first_name']} {row['last_name']}", 
                fontsize=8, ha='center', color='black')

    # Show the plot
    plt.show()


def main():

    pop = statcast_catcher_poptime(2024, 0, 0)
    framing = statcast_catcher_framing(2024, 6)

    # Get player name input
    user_in = input("Enter player: ")

    # Get player index for framing
    framing_player_index = get_player_index(user_in, framing)
    if framing_player_index is None:
        print("Could not find player in framing data! Please try again.")
        return

    # Get player index for poptime
    player_name = f"{framing.iloc[framing_player_index].first_name} {framing.iloc[framing_player_index].last_name}"
    pop_player_index = get_pop_player_index(player_name, pop)
    if pop_player_index is None:
        print("Could not find player in poptime data! Please try again.")
        return

    # Get catcher framing data
    catcher_framing = framing.iloc[framing_player_index]
    catcher_pop = pop.iloc[pop_player_index]

    player_id = str(int(framing.iloc[framing_player_index].player_id)) #why tf are player id's float values?
    img = get_player_headshot(player_id)

    #x = pybaseball.fielding_stats(2024)

    all_catcher_ids = collect_player_ids(framing)

    all_catcher_throwing = get_all_catcher_throwing(all_catcher_ids)

    catcher_throwing = get_catcher_throwing(str(player_id), '2024')

    csaa = get_csaa(catcher_throwing)

    print(f"CSAA: {csaa[0]} CSAA no teamwork: {csaa[1]} CSAA no pitcher: {csaa[2]}")

    all_csaa = get_all_csaa(all_catcher_throwing)
        

    # Create mappings for each stat
    csaa_percentile_mappings = {col: get_percentile_mapping(all_csaa[col]) for col in all_csaa}    
    pop_percentile_mappings = {col: get_percentile_mapping(pop[col]) for col in pop.columns}
    framing_percentile_mappings = {col: get_percentile_mapping(framing[col]) for col in framing.columns}

    player_data = statsapi.player_stat_data(player_id, group="[fielding]", type="yearByYear")

    stats23 = get_catcher_stats(player_data, '2023') or None
    stats24 = get_catcher_stats(player_data, '2024') or None

    current_team = get_current_team(player_data)

    # Function to get the grid color for a specific strike rate
    def get_grid_color(key):
        percentile = find_percentile(catcher_framing[key], framing_percentile_mappings[key])
        
        if percentile < 40:
            return {'color': 'crimson', 'value': f'{percentile}'}  # Correct string formatting
        if percentile < 55:
            return {'color': 'yellow', 'value': f'{percentile}'}  # Correct string formatting
        if percentile <= 70:
            return {'color': 'lightgreen', 'value': f'{percentile}'}  # Correct string formatting
        
        return {'color': 'green', 'value': f'{percentile}'}  # Corrected return with dictionary

    # Create the figure with a custom layout
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter root window

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Create the figure and plot
    fig = plt.figure(figsize=(screen_width / 100, screen_height / 100))  # Screen size in inches (100 dpi)
    fig.subplots_adjust(hspace=0.25)  # Increase vertical spacing between rows

    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])  # 2 rows, 3 columns

    splot = plt.figure(figsize=(screen_width / 100, screen_height / 100))

    # Upper Left: Placeholder Bar Graph
    ax1 = fig.add_subplot(gs[0, 0])
    create_bar_plot(ax1)

    # Upper Middle: Headshot and General Information
    ax2 = fig.add_subplot(gs[0, 1])
    create_headshot_info_plot(ax2, csaa, img)

    # Upper Right: Logo
    ax3 = fig.add_subplot(gs[0, 2])
    show_logo(ax3)

    # Bottom Left: Placeholder Bar Graph
    ax4 = fig.add_subplot(gs[1, 0])
    create_csaa_plot(ax4, csaa, csaa_percentile_mappings)

    # Bottom Center: Strike Zone Plot 
    ax5 = fig.add_subplot(gs[1, 1])
    create_strike_zone_plot(ax5, catcher_framing, get_grid_color)

    #Bottom Right: Key stats
    ax6 = fig.add_subplot(gs[1, 2])
    
    create_key_stats_plot(ax6, stats23, stats24)

    measurables = statsapi.get('person', {'personId':str(player_id)})['people'][0]
    measurables = f"Age: {measurables.get('currentAge')}     Ht: {measurables.get('height')}     Wt: {measurables.get('weight')}"

    #arbitrarily place titles
    fig.text(0.5125, 0.08, "Shadow Zone Strike Rate", ha='center', va='center', fontsize=12, fontweight="bold") # strike zone 'title'
    fig.text(0.5125, 0.05, "& Framing Runs", ha='center', va='center', fontsize=12, fontweight="bold") # strike zone 'title'
    fig.text(0.5125, 0.95, f"{player_name}", ha='center', va='center', fontsize=12, fontweight="bold") # player name
    fig.text(0.5125, 0.9, f"{current_team}", ha='center', va='center', fontsize=12) 
    fig.text(.51, .5, f"{measurables}", ha='center', va='center', fontsize=10) 

    # Adjust layout
    plt.tight_layout() 

    # Show the plot
    #plt.show()

    all_csaa = all_csaa.merge(framing[['player_id', 'first_name', 'last_name']], on='player_id', how='left')
    
    create_csaa_scatter_plot(all_csaa)

    #will_smith_id = 0
    #will_smith_data = get_catcher_throwing(will_smith_id, '2024')
    #print(will_smith_data)


if __name__ == "__main__":
    main()
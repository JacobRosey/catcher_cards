from pybaseball import statcast_catcher_framing, statcast_catcher_poptime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import requests
from PIL import Image
import requests
from io import BytesIO
import tkinter as tk
import statsapi
import os
import ast
import unicodedata

# "Christian Vazquez" == "Christian Vázquez"
def normalize_text(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'
    )

def are_equivalent(text1, text2):
    return normalize_text(text1).lower() == normalize_text(text2).lower()

def get_framing_player_index(user_in, unfiltered_framing):
    for index, row in unfiltered_framing.iterrows():
        if index == 0:  # Skip league average row
            continue
        if are_equivalent(user_in, row.first_name + ' ' + row.last_name):
            return index
        print(row.first_name + " " + row.last_name)
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
    percentiles = np.arange(0, 101, 1)  # 0th to 100th percentiles, steps of 5
    values = np.percentile(column.dropna(), percentiles)
    
    return dict(zip(percentiles, values))

# Function to find the percentile of a value
def find_percentile(value, mapping):
    for percentile, threshold in sorted(mapping.items()):
        if value <= threshold:
            return percentile
    return 100

def add_suffix(percentile):
    percentileAsStr = str(percentile)
    last_digit = int(percentileAsStr[-1])  # Convert last digit to int

    if 11 <= percentile <= 13:  # Handle special cases (11th, 12th, 13th)
        suffix = "th"
    elif last_digit == 1:
        suffix = "st"
    elif last_digit == 2:
        suffix = "nd"
    elif last_digit == 3:
        suffix = "rd"
    else:
        suffix = "th"

    return f"{percentile}{suffix}"

# Function to get the grid color for a percentile
def get_percentile_info(data, key, mapping):
    percentile = find_percentile(data[key], mapping[key])

    if percentile < 40:
        color = "crimson"
    elif percentile < 55:
        color = "yellow"
    elif percentile <= 70:
        color = "lightgreen"
    else:
        color = "green"

    return {'color': color, 'value': percentile}  # Keep percentile as int

# Used to calculate percentiles where lower numbers are better
def invert_percentile(item):
    item['value'] = 100 - item['value']  # Keep it an integer
    
    # Color mapping using a dictionary
    color_mapping = {
        'crimson': 'green',
        'green': 'crimson',
        'lightgreen': 'yellow',
        'yellow': 'lightgreen'
    }
    
    # Get the new color
    item['color'] = color_mapping.get(item['color'], item['color'])
    
    return item

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

def get_catcher_blocking(player_id, year):
    filename = f"data/{player_id}_{year}_blocking.csv"
    if os.path.exists(filename):
        print(f"File {filename} exists. Returning the data from the file.")
        return pd.read_csv(filename)
    
    url = f'https://baseballsavant.mlb.com/leaderboard/services/catcher-blocking/{player_id}?game_type=Regular&n=q&season_end={year}&season_start={year}&split=no&team=&type=Cat&with_team_only=1'
 
    # Send a GET request to the leaderboard page
    response = requests.get(url)
    
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        df.to_csv(filename, index = False, header = True)
        return df
    else:
        print(f"Error: {response.status_code}")
        return None
     
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

def calculate_blocking_metrics(df):
    # Check if the 'data' column exists
    if 'data' not in df.columns:
        raise ValueError("Data column not found in the DataFrame.")
    
    # Apply ast.literal_eval only if 'data' column is stringified (i.e., read from CSV)
    if isinstance(df['data'].iloc[0], str):
        # Convert stringified dictionaries into actual dictionaries
        df['data'] = df['data'].apply(ast.literal_eval)
    
    # Normalize the 'data' column (expand it into multiple columns)
    df_expanded = pd.json_normalize(df['data'])

    # Merge the expanded columns back into the original DataFrame
    df = df.join(df_expanded)

    # Check if necessary columns exist
    if not {'x_runner_pbwp', 'is_runner_pbwp'}.issubset(df.columns):
        print("Columns in DataFrame:", df.columns)
        raise ValueError("DataFrame must contain 'x_runner_pbwp' and 'is_runner_pbwp' columns.")
    
    # Apply categorization for difficulty
    def categorize_difficulty(x_pbwp):
        if x_pbwp < 0.05:
            return 'Easy'
        elif 0.05 <= x_pbwp < 0.15:
            return 'Medium'
        else:
            return 'Hard'

    df['difficulty'] = df['x_runner_pbwp'].apply(categorize_difficulty)

    # Group by difficulty and calculate metrics
    results = []
    for difficulty, group in df.groupby('difficulty'):
        total_opportunities = len(group)
        total_passed_balls = group['is_runner_pbwp'].sum()
        expected_passed_balls = group['x_runner_pbwp'].sum()

        block_percent = 1 - (total_passed_balls / total_opportunities) if total_opportunities > 0 else 0
        blocks_above_average = (total_opportunities - total_passed_balls) - (total_opportunities - expected_passed_balls)

        results.append({
            'Difficulty': difficulty,
            'Total Opportunities': total_opportunities,
            'Total Passed Balls': total_passed_balls,
            'Expected Passed Balls': expected_passed_balls,
            'Block %': block_percent,
            'Blocks Above Average': blocks_above_average
        })

    return pd.DataFrame(results)

def create_metrics_table(ax, df):
    # Rename columns
    df = df.rename(columns={
        'Difficulty': 'Difficulty',
        'Block %': '% Blocked',
        'Total Passed Balls': 'PBWP',
        'Expected Passed Balls': 'xPBWP',
        'Blocks Above Average': 'BAA'
    })

    # Remove the 'Total Opportunities' column
    df = df.drop(columns=['Total Opportunities'])

    # Need to strip .0 from this int somehow !!!!!!!!!!!!
    df['PBWP'] = df['PBWP'].astype(int).astype(str)

    # Round to 2 decimal places
    df[['xPBWP', 'BAA']] = df[['xPBWP', 'BAA']].round(2)

    df['% Blocked'] = (df['% Blocked'].round(3) * 100).round(2)

    # Define the custom order for 'Difficulty' and sort accordingly
    difficulty_order = ['Easy', 'Medium', 'Hard']
    df['Difficulty'] = pd.Categorical(df['Difficulty'], categories=difficulty_order, ordered=True)
    
    df = df[['Difficulty', '% Blocked', 'PBWP', 'xPBWP', 'BAA']]

    # Sort by difficulty
    df = df.sort_values(by='Difficulty')

    # Pivot the dataframe to have difficulty levels as columns
    df_pivot = df.set_index('Difficulty').transpose()

    # Create the table
    table = ax.table(cellText=df_pivot.values, colLabels=df_pivot.columns, rowLabels=df_pivot.index, loc='center', cellLoc='center', bbox=[0.25, 0.1, 0.75, 0.8])

    # Set table styling
    for key, cell in table.get_celld().items():
        cell.set_height(0.18)  # Adjust cell height
        cell.set_width(0.25)   # Adjust cell width

    # Highlight the header row and column
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:  # Header row and first column
            cell.set_text_props(weight='bold', color='black', ha='center')
            cell.set_facecolor('#e0e0e0') 
        else: 
            cell.set_text_props(color='black', ha='center')
            cell.set_facecolor('#ffffff') 


    total_baa = (df['xPBWP'].astype(float) - df['PBWP'].astype(int)).sum()
    ax.text(0.49, 0, f"Total Blocks Above Average: {total_baa:.1f}", ha='center', va='center', fontsize=9, color='black', transform=ax.transAxes, fontweight='bold')
    
    # Remove axes for better visibility
    ax.axis('off')
    ax.set_title("Blocks Above Average", fontweight='bold')

def create_strike_zone_plot(ax, catcher_framing, framing_percentile_mappings):
    # Draw the strike zone grid
    ax.add_patch(patches.Rectangle((0, 0), 3, 3, edgecolor='black', facecolor='none'))
    ax.add_patch(patches.Rectangle((1, 1), 1, 1, edgecolor='black', facecolor='none'))

    ax.add_patch(patches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_17', framing_percentile_mappings)['color']))
    ax.add_patch(patches.Rectangle((2, 0), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_19', framing_percentile_mappings)['color']))
    ax.add_patch(patches.Rectangle((0, 2), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_11', framing_percentile_mappings)['color']))
    ax.add_patch(patches.Rectangle((2, 2), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_13', framing_percentile_mappings)['color']))

    ax.add_patch(patches.Rectangle((1, 0), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_18', framing_percentile_mappings)['color']))
    ax.add_patch(patches.Rectangle((1, 2), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_12', framing_percentile_mappings)['color']))
    ax.add_patch(patches.Rectangle((0, 1), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_14', framing_percentile_mappings)['color']))
    ax.add_patch(patches.Rectangle((2, 1), 1, 1, edgecolor='black', facecolor=get_percentile_info(catcher_framing, 'strike_rate_16', framing_percentile_mappings)['color']))

    # Set axis limits and labels
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes

    # Add labels
    ax.text(0.5, 2.5, f"{catcher_framing.strike_rate_11}%", ha='center', va='center', fontsize=10)
    ax.text(0.5, 2.2,  f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_11', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(1.5, 2.5, f"{catcher_framing.strike_rate_12}%", ha='center', va='center', fontsize=10)
    ax.text(1.5, 2.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_12', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(2.5, 2.5, f"{catcher_framing.strike_rate_13}%", ha='center', va='center', fontsize=10)
    ax.text(2.5, 2.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_13', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(0.5, 1.5, f"{catcher_framing.strike_rate_14}%", ha='center', va='center', fontsize=10)
    ax.text(0.5, 1.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_14', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(1.5, 1.5, f"FR: {catcher_framing.runs_extra_strikes}", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 1.2, f"{add_suffix(get_percentile_info(catcher_framing, 'runs_extra_strikes', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(2.5, 1.5, f"{catcher_framing.strike_rate_16}%", ha='center', va='center', fontsize=10)
    ax.text(2.5, 1.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_16', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(0.5, 0.5, f"{catcher_framing.strike_rate_17}%", ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_17', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(1.5, 0.5, f"{catcher_framing.strike_rate_18}%", ha='center', va='center', fontsize=10)
    ax.text(1.5, 0.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_18', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)
    ax.text(2.5, 0.5, f"{catcher_framing.strike_rate_19}%", ha='center', va='center', fontsize=10)
    ax.text(2.5, 0.2, f"{add_suffix(get_percentile_info(catcher_framing, 'strike_rate_19', framing_percentile_mappings)['value'])}", ha='center', va='center', fontsize=8)

def create_csaa_plot(ax, csaa, csaa_percentile_mappings):
    csaa = pd.DataFrame([csaa], columns =['csaa', 'csaa_no_teamwork', 'pure_csaa'])  # Let pandas infer column names properly
    
    categories = ['CSAA', 'Pure CSAA']
    
    # Retrieve percentile info
    csaa_info = get_percentile_info(csaa.iloc[0], 'csaa', csaa_percentile_mappings)
    pure_csaa_info = get_percentile_info(csaa.iloc[0], 'pure_csaa', csaa_percentile_mappings)
    
    # Ensure CSAA is first, Pure CSAA second
    percentiles = [csaa_info['value'], pure_csaa_info['value']]
    values = [csaa.iloc[0]['csaa'], csaa.iloc[0]['pure_csaa']]
    colors = [csaa_info['color'], pure_csaa_info['color']]
    
    # Create vertical bar chart (percentiles on y-axis)
    ax.bar(categories, percentiles, color=colors, width=.75)
    ax.set_ylim(0, 100)  # Percentiles range from 0 to 100
    ax.set_ylabel('Percentile')

    # Add values on the bars
    for i, (value, percentile) in enumerate(zip(values, percentiles)):
        offset = 0 if percentile <= 10 else -10  # Adjust text positioning
        ax.text(i, percentile + offset, f'{value:.2f}', va='bottom', fontsize=10, ha='center')
    
    # Remove unnecessary spines and format the plot
    ax.set_title('CSAA & Pure CSAA', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return ax

    #ax.set_title("CSAA & Pure CSAA", fontweight='bold')

def create_pop_plot(ax, stats, pop_percentile_mappings):
    
    # Get percentile info for each stat - invert those where smaller numbers are better
    exchange = invert_percentile(get_percentile_info(stats, 'exchange_2b_3b_sba', pop_percentile_mappings))
    pop2b = invert_percentile(get_percentile_info(stats, 'pop_2b_sba', pop_percentile_mappings))
    pop3b = invert_percentile(get_percentile_info(stats, 'pop_3b_sba', pop_percentile_mappings))
    maxeff_velo = get_percentile_info(stats, 'maxeff_arm_2b_3b_sba', pop_percentile_mappings)

    # Prepare categories and values for the bar chart
    categories = ['Exchange', 'Pop 2B', 'Pop 3B', 'Max Eff Velo']
    percentiles = [int(exchange['value']), int(pop2b['value']), int(pop3b['value']), int(maxeff_velo['value'])]
    values = [float(stats['exchange_2b_3b_sba']), float(stats['pop_2b_sba']), float(stats['pop_3b_sba']), float(stats['maxeff_arm_2b_3b_sba'])]
    colors = [exchange['color'], pop2b['color'], pop3b['color'], maxeff_velo['color']]
    
    # Create the vertical bar chart
    ax.bar(categories, percentiles, color=colors)
    ax.set_ylim(0, 100)  # Percentiles are between 0 and 100
    ax.set_ylabel('Percentile')
    
    # Add values on the bars
    for i, (value, percentile) in enumerate(zip(values, percentiles)):
        offset = 0 if percentile <= 10 else -10
        ax.text(i, percentile + offset, f'{value:.2f}', va='bottom', fontsize=10, ha='center')
    
    # Remove axis and set labels
    ax.set_title('Pop Time & Exchange', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)  

def create_headshot_info_plot(ax, player_name, img):
    ax.imshow(img, aspect='auto', extent=[0.2, 0.3, 0.4, 0.1])  # Adjust extent to position the image
    ax.text(0.5, 1.0, f"{player_name}", ha='center', va='center', fontsize=12)
    ax.axis('off')  # Hide axes

def create_key_stats_plot(ax, stats23, stats24):
    cell_text = [
        ["", "2023", "2024"],
        ["Innings", stats23.get("innings", "No data"), stats24.get("innings", "No data")],
        ["Fielding %", stats23.get("fielding", "No data"), stats24.get("fielding", "No data")],
        ["E", stats23.get("errors", "No data"), stats24.get("errors", "No data")],
        #["DP", stats23.get("doublePlays", "No data"), stats24.get("doublePlays", "No data")],
        ["PB", stats23.get("passedBall", "No data"), stats24.get("passedBall", "No data")],
        ["WP", stats23.get("wildPitches", "No data"), stats24.get("wildPitches", "No data")],
        #["CERA", stats23.get("catcherERA", "No data"), stats24.get("catcherERA", "No data")]
    ]

    table = ax.table(cellText=cell_text, loc='center', cellLoc='center')

    for key, cell in table.get_celld().items():
        cell.set_height(.18)  
        cell.set_width(.32)   

    # Highlight the header row
    for (row, col), cell in table.get_celld().items():
        if row == 0 and col == 0:
            cell.set_linewidth(0)
            cell.set_facecolor('none')
            continue
        if row == 0 or col == 0:  # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#e0e0e0')  # Light gray background for header
           
    ax.axis('off')

def get_catcher_stats(player_data, year):
    # Find the stats for year where the position is "Catcher"
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


def get_all_catcher_throwing(ids, filename='data/catcher_throwing_data.csv'):
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
    
    return all_catcher_throwing 


def get_csaa(catcher_throwing):
    csaa = 0
    teamwork = 0
    pitching = 0
    total_cs = 0
    total_sb = 0

    for _, row in catcher_throwing.iterrows():
        data = row.get('data', [])  
        csaa += data.get('cs_aa', 0)
        teamwork += data.get('teamwork_over_xcs', 0)
        if data.get('is_runner_cs') == 1:
            total_cs +=1
        if data.get('is_runner_sb') == 1:
            total_sb +=1
        if isinstance(data.get('pitcher_cs_aa'), float):  
            pitching += data['pitcher_cs_aa']
    rSB = ((csaa * .65))
    print(f"steals: {total_sb}")
    print(f"caught stealing; {total_cs}")
    print(rSB)
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
    
    # Filter out players whose CSAA is too close to the origin (near 0)
    all_csaa_filtered = all_csaa[
    ~((all_csaa['csaa'].abs() <= 4) | (all_csaa['csaa_difference'].abs() < 2) | 
      (all_csaa['last_name'] == "Knizner") | (all_csaa['last_name'] == "Ruiz"))
]

    # Specific additions by request
    aw_row = all_csaa[(all_csaa['first_name'] == 'Austin') & (all_csaa['last_name'] == 'Wells')]
    wc_row = all_csaa[(all_csaa['first_name'] == 'William') & (all_csaa['last_name'] == 'Contreras')]
    yd_row = all_csaa[(all_csaa['first_name'] == 'Yainer') & (all_csaa['last_name'] == 'Diaz')]
    fa_row = all_csaa[(all_csaa['first_name'] == 'Francisco') & (all_csaa['last_name'] == 'Alvarez')]
    all_csaa_filtered = pd.concat([all_csaa_filtered, aw_row, wc_row, yd_row, fa_row])

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

    return plt

def find_best_caught_stealing(df, player_name):
   
    if 'data' in df.columns:
        df = pd.json_normalize(df['data'])

    # Find plays where runner is thrown out when expected to be safe
    filtered_df = df[(df['is_runner_cs'] == 1) & (df['exp_cs'] <= 0.5)]

    filtered_df['play_url'] = filtered_df['play_id'].apply(
        lambda play_id: f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    )

    filtered_df = filtered_df.sort_values(by='exp_cs', ascending=True)

    # Save the filtered DataFrame to a CSV file
    filtered_df['play_url'].to_csv(f"video_links/best_cs/{player_name}.csv", index=False)

def find_worst_caught_stealing(df, player_name):
    
    if 'data' in df.columns:
        df = pd.json_normalize(df['data'])

    # Find plays where runner was safe when expected to be thrown out
    filtered_df = df[(df['is_runner_cs'] == 0) & (df['exp_cs'] >= 0.5)]

    filtered_df['play_url'] = filtered_df['play_id'].apply(
        lambda play_id: f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    )

    filtered_df = filtered_df.sort_values(by='exp_cs', ascending=False)

    # Save the filtered DataFrame to a CSV file
    filtered_df['play_url'].to_csv(f"video_links/worst_cs/{player_name}.csv", index=False)



def main():

    pop = statcast_catcher_poptime(2024, 0, 0)
    framing = statcast_catcher_framing(2024, 6)

    # Get player name input
    user_in = input("Enter player: ")

    # Get player index for framing
    framing_player_index = get_framing_player_index(user_in, framing)
    if framing_player_index is None:
        print("Could not find player in framing data! Please try again.")
        return

    # Get player index for poptime
    player_name = f"{framing.iloc[framing_player_index].first_name} {framing.iloc[framing_player_index].last_name}"
    pop_player_index = get_pop_player_index(player_name, pop)
    if pop_player_index is None:
        print("Could not find player in poptime data! Please try again.")
        return


    player_id = str(int(framing.iloc[framing_player_index].player_id))
    img = get_player_headshot(player_id)

    all_catcher_ids = collect_player_ids(framing)

    all_catcher_throwing = get_all_catcher_throwing(all_catcher_ids)

    catcher_throwing = get_catcher_throwing(str(player_id), '2024')

    csaa = get_csaa(catcher_throwing)

    all_csaa = get_all_csaa(all_catcher_throwing)

    # Create mappings for each stat
    csaa_percentile_mappings = {col: get_percentile_mapping(all_csaa[col]) for col in all_csaa}    
    pop_percentile_mappings = {col: get_percentile_mapping(pop[col]) for col in pop.columns}
    framing_percentile_mappings = {col: get_percentile_mapping(framing[col]) for col in framing.columns}
  
    player_data = statsapi.player_stat_data(player_id, group="[fielding]", type="yearByYear")

    stats23 = get_catcher_stats(player_data, '2023') or None
    stats24 = get_catcher_stats(player_data, '2024') or None

    current_team = get_current_team(player_data)

    catcher_blocking = calculate_blocking_metrics(get_catcher_blocking(str(player_id), '2024'))

    # Create the figure with a custom layout
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter root window

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Create the figure and plot
    fig = plt.figure(figsize=(screen_width / 100, screen_height / 100), facecolor="#F0F0F0")  # Screen size in inches (100 dpi)
    fig.subplots_adjust(hspace=.35)  # Increase vertical spacing between rows

    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])  # 2 rows, 3 columns

    # Upper Left: Pop time and exchange
    ax1 = fig.add_subplot(gs[0, 0])
    create_pop_plot(ax1, pop.iloc[pop_player_index], pop_percentile_mappings)

    # Upper Middle: Headshot and General Information
    ax2 = fig.add_subplot(gs[0, 1])
    create_headshot_info_plot(ax2, csaa, img)

    # Upper Right: blocking
    ax3 = fig.add_subplot(gs[0, 2])
    create_metrics_table(ax3, catcher_blocking)

    # Bottom Left: csaa
    ax4 = fig.add_subplot(gs[1, 0])
    create_csaa_plot(ax4, csaa, csaa_percentile_mappings)

    # Bottom Center: Strike Zone Plot 
    ax5 = fig.add_subplot(gs[1, 1])
    create_strike_zone_plot(ax5, framing.iloc[framing_player_index], framing_percentile_mappings)

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
    fig.text(.51, .52, f"{measurables}", ha='center', va='center', fontsize=10) 
    fig.text(.51, .48, "@kickdirtbb on X.com", ha='center', va='center', fontsize=9) 


    #video plays by id: https://baseballsavant.mlb.com/sporty-videos?playId={}
    find_best_caught_stealing(catcher_throwing, player_name)
    find_worst_caught_stealing(catcher_throwing, player_name)
   
    # Adjust layout
    plt.tight_layout() 

    # Show the main graphic
    #plt.show()
    full_path = f"cards/{player_name}_2024"
    plt.savefig(full_path)

    # Get the scatterplot
    #csaa_plot = create_csaa_scatter_plot(all_csaa)

    # Show the scatterplot
    #csaa_plot.show()

if __name__ == "__main__":
    main()

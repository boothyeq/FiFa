import pandas as pd

# Set the maximum column width to a larger value (e.g., 200)
pd.set_option('display.max_colwidth', 200)

# Load the CSV file
df = pd.read_csv(r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv')

position_mapping = {
    'CF': 'Center Forward',
    'RW': 'Right Winger',
    'ST': 'Striker',
    'CAM': 'Central Attacking Midfielder',
    'RM': 'Right Midfielder',
    'CM': 'Central Midfielder',
    'LW': 'Left Winger',
    'CB': 'Center Back',
    'GK': 'Goalkeeper',
    'CDM': 'Central Defensive Midfielder',
    'LB': 'Left Back',
    'LM': 'Left Midfielder',
    'RB': 'Right Back',
    'RWB': 'Right Wing Back',
    'LWB': 'Left Wing Back'
}

if 'positions' in df.columns:
    def convert_to_long_form(short_positions):
        if isinstance(short_positions, str):
            return ', '.join(position_mapping.get(pos.strip(), pos.strip()) for pos in short_positions.split(','))
        return short_positions

    df['long_form_positions'] = df['positions'].apply(convert_to_long_form)

    print("First 5 Rows with Original and Long-Form Positions:")
    print(df[['positions', 'long_form_positions']].head())

else:
    print("The 'positions' column does not exist in the dataset.")
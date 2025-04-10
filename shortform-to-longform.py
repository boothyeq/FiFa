import pandas as pd
import os  # Import the os module

# Load the CSV file
file_path = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'
df = pd.read_csv(file_path)

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

print(f"Checking if 'positions' column exists: {'positions' in df.columns}")

if 'positions' in df.columns:
    def convert_to_long_form(short_positions):
        if isinstance(short_positions, str):
            return ', '.join(position_mapping.get(pos.strip(), pos.strip()) for pos in short_positions.split(','))
        return short_positions

    # Replace the original 'positions' column with the long-form values
    df['positions'] = df['positions'].apply(convert_to_long_form)

    # List of columns to remove
    columns_to_remove = ['nationality', 'national_team_position', 'birth_date', 'preferred_foot', 'body_type', 'national_jersey_number']

    # Print the column names in the DataFrame before dropping
    print("\nColumn names in the DataFrame before dropping:")
    print(df.columns.tolist())
    print("\nColumns to remove:")
    print(columns_to_remove)

    # Drop the specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')
    # The 'errors='ignore' argument will prevent an error if any of the columns are not found.

    # Define the output file path
    output_dir = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE'
    output_file_name = 'fifa_players_cleaned.csv'
    output_file_path = os.path.join(output_dir, output_file_name)

    print(f"\nAttempting to save to: {output_file_path}")

    try:
        df.to_csv(output_file_path, index=False)
        print(f"New CSV file created successfully at: {output_file_path}")
    except Exception as e:
        print(f"An error occurred during saving: {e}")

else:
    print("The 'positions' column does not exist in the dataset.")
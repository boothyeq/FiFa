import pandas as pd
import os

# ONLY REMOVING THE UNWANTED COLUMNS AND CHANGING TO LONG FORM

# Load the CSV file
file_path = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'
df = pd.read_csv(file_path)

# print sample of the 'positions' column BEFORE conversion
print("BEFORE Long-Form Conversion (First 5 rows):\n", df['positions'].head())

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

    # do conversion
    df['long_form_positions'] = df['positions'].apply(convert_to_long_form)

    # primt sample of the original and new 'positions' columns AFTER conversion
    print("\nAFTER Long-Form Conversion (First 5 rows):\n", df[['positions', 'long_form_positions']].head())

    # List columns 2 remove
    columns_to_remove = ['nationality', 'national_team_position', 'birth_date', 'preferred_foot', 'body_type', 'national_jersey_number']

    # print column names BEFORE dropping
    print("\nColumns BEFORE dropping:\n", df.columns.tolist())

    # Drop the specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')

    # print column names AFTER dropping
    print("\nColumns AFTER dropping:\n", df.columns.tolist())

    # # d output file path
    # output_dir = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE'
    # output_file_name = 'fifa_players_cleaned.csv'
    # output_file_path = os.path.join(output_dir, output_file_name)
    #
    # try:
    #     df.to_csv(output_file_path, index=False)
    #     print(f"\nNew CSV file created successfully at: {output_file_path}")
    # except Exception as e:
    #     print(f"\nAn error occurred during saving: {e}")

else:
    print("The 'positions' column does not exist in the dataset.")
import pandas as pd

# Load the CSV file
file_path = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'
df = pd.read_csv(file_path)

# Select columns from the beginning up to (but not including) the 3rd column (index 2)
# and then select columns from the 4th column (index 3) up to (but not including) the 6th column (index 5)
columns_to_keep = df.columns[:2].tolist() + df.columns[3:5].tolist()

# Create a new DataFrame with only the selected columns
df_selected = df[columns_to_keep]

for column in df.columns:
        print(column)

print("DataFrame with the first two and the fourth and fifth columns:\n")
print(df_selected.head())
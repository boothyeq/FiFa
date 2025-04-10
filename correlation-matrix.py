import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'

try:
    df = pd.read_csv(file_path)

    # Select only the numerical columns for the correlation matrix
    numerical_df = df[[
        'age', 'height_cm', 'weight_kgs', 'overall_rating', 'potential', 'value_euro',
        'wage_euro', 'international_reputation(1-5)', 'weak_foot(1-5)', 'skill_moves(1-5)',
        'release_clause_euro', 'national_rating', 'national_jersey_number', 'crossing',
        'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve',
        'freekick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
        'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength',
        'long_shots', 'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
        'composure', 'marking', 'standing_tackle', 'sliding_tackle'
    ]]

    # Calculate the correlation matrix
    correlation_matrix = numerical_df.corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(20, 18))  # Adjust figure size for better readability
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical FIFA Player Features')
    plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
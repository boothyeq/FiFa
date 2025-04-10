import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'

try:
    df = pd.read_csv(file_path)

    # Limit to the top N nationalities to avoid a cluttered plot
    top_n = 15
    top_nationalities = df['nationality'].value_counts().nlargest(top_n).index
    df_top_n_nationalities = df[df['nationality'].isin(top_nationalities)]

    # Create the box plot for wage
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='nationality', y='wage_euro', data=df_top_n_nationalities)
    plt.xlabel('Nationality')
    plt.ylabel('Wage (€)')
    plt.title(f'Distribution of Player Wage (€) by Top {top_n} Nationalities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {file_path}")
except KeyError:
    print(f"Error: One or both of the specified columns ('nationality', 'wage_euro') were not found in the CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")
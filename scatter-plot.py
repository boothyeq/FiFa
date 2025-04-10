import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'

try:
    df = pd.read_csv(file_path)

    # Specify the columns for the scatter plot
    x_column = 'overall_rating'
    y_column = 'potential'

    # Create colors based on potential
    colors = []
    for potential in df['potential']:
        if potential < 50:
            colors.append('red')
        elif 50 <= potential <= 60:
            colors.append('orange')
        elif 60 < potential <= 70:
            colors.append('yellow')
        elif 70 < potential <= 90:
            colors.append('lightgreen')
        else:  # potential > 90
            colors.append('lime')  # Using 'lime' for bright green

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], c=colors, alpha=0.6)

    # Add labels and title
    plt.xlabel('Overall Rating')
    plt.ylabel('Potential')
    plt.title('Scatter Plot of Potential vs Overall Rating (Colored by Potential)')

    # Add a grid (optional)
    plt.grid(True)

    # Add a legend
    legend_elements = [plt.scatter([], [], color='red', label='Potential < 50', alpha=0.6),
                       plt.scatter([], [], color='orange', label='50 <= Potential <= 60', alpha=0.6),
                       plt.scatter([], [], color='yellow', label='60 < Potential <= 70', alpha=0.6),
                       plt.scatter([], [], color='lightgreen', label='70 < Potential <= 90', alpha=0.6),
                       plt.scatter([], [], color='lime', label='Potential > 90', alpha=0.6)]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {file_path}")
except KeyError:
    print(f"Error: One or both of the specified columns ('{x_column}', '{y_column}', 'potential') were not found in the CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")
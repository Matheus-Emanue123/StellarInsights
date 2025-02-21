# Import required libraries

import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Function: Load dataset
def load_data(file_path, earth_like_indices):
    # Load the dataset, skipping comments
    data = pd.read_csv("../databases/PS_2025.01.12_10.15.56.csv", comment='#', low_memory=False)
    
    # Separate Earth-like planets
    valid_indices = [idx for idx in earth_like_indices if idx in data.index]
    earth_like = data.loc[valid_indices]

    # Add Earth and Similar Earth manually
    earth_reference = {
        "st_teff": 5778,
        "st_rad": 1.0,
        "st_mass": 1.0,
        "pl_insol": 1.0,
        "pl_eqt": 288.0,
        "pl_orbper": 365.25,
        "pl_orbsmax": 1.0,
        "pl_orbeccen": 0.0167,
        "pl_dens": 5.51,
        "st_met": 0.0,
        "planet_name": "Earth",
        "index": 38047
    }

    similar_earth_reference = {
        "st_teff": 5778,
        "st_rad": 1.0,
        "st_mass": 1.0,
        "pl_insol": 1.02,
        "pl_eqt": 290.0,
        "pl_orbper": 365.0,
        "pl_orbsmax": 1.01,
        "pl_orbeccen": 0.02,
        "pl_dens": 5.5,
        "st_met": 0.01,
        "planet_name": "Similar Earth",
        "index": 38048
    }

    # Convert to DataFrame and append
    artificial_entries = pd.DataFrame([earth_reference, similar_earth_reference])
    artificial_entries.set_index("index", inplace=True)
    earth_like = pd.concat([earth_like, artificial_entries])

    # All other stars (exclude Earth-like indices)
    all_other_stars = data.loc[~data.index.isin(valid_indices)]

    return data, earth_like, all_other_stars

# Function: Analyze metallicity
def analyze_metallicity(earth_like, all_other_stars):
    sns.set_theme(style="whitegrid", palette="deep")
    
    # Extract metallicity data
    earth_like_metallicity = earth_like['st_met']
    other_stars_metallicity = all_other_stars['st_met']

    # Plot comparison
    plt.figure(figsize=(12, 6))
    sns.kdeplot(earth_like_metallicity, color='blue', label='Earth-like Planets', fill=True, alpha=0.6, linewidth=2)
    sns.kdeplot(other_stars_metallicity, color='red', label='Other Stars', fill=True, alpha=0.6, linewidth=2)
    plt.xlabel("Metallicity [Fe/H]", fontsize=14, fontweight="bold")
    plt.ylabel("Density", fontsize=14, fontweight="bold")
    plt.title("Comparison of Metallicity between Earth-like Planets and Other Stars", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Perform a t-test
    stat, p_value = ttest_ind(earth_like_metallicity.dropna(), other_stars_metallicity.dropna())
    print(f"Metallicity - t-statistic: {stat}, p-value: {p_value}")

# Function: Analyze stellar mass
def analyze_stellar_mass(earth_like, all_other_stars):
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Extract stellar mass data
    earth_like_mass = earth_like['st_mass']
    other_stars_mass = all_other_stars['st_mass']

    # Plot comparison
    plt.figure(figsize=(12, 6))
    sns.kdeplot(earth_like_mass, color='green', label='Earth-like Planets', fill=True, alpha=0.6, linewidth=2)
    sns.kdeplot(other_stars_mass, color='orange', label='Other Stars', fill=True, alpha=0.6, linewidth=2)
    plt.xlabel("Stellar Mass [Solar Mass]", fontsize=14, fontweight="bold")
    plt.ylabel("Density", fontsize=14, fontweight="bold")
    plt.title("Comparison of Stellar Mass between Earth-like Planets and Other Stars", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Perform a t-test
    stat, p_value = ttest_ind(earth_like_mass.dropna(), other_stars_mass.dropna())
    print(f"Stellar Mass - t-statistic: {stat}, p-value: {p_value}")

# Function: Identify outliers
def identify_outliers(earth_like, all_other_stars):
    # Define criteria for outliers
    outliers = all_other_stars[
        (all_other_stars['st_met'] > 1.5) | 
        (all_other_stars['st_met'] < -1.5) |
        (all_other_stars['st_mass'] > 3)
    ]
    
    # Print the outliers
    print("Outliers identified:")
    if 'planet_name' in outliers.columns:
        print(outliers[['st_met', 'st_mass', 'planet_name']])
    else:
        print(outliers[['st_met', 'st_mass']])
    
    return outliers

# Function: Combined analysis (scatter plot with solar lines and annotations)
def combined_analysis_with_outliers(earth_like, all_other_stars, outliers):
    sns.set_theme(style="white", palette="pastel")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x='st_met', y='st_mass', data=all_other_stars,
        color='gray', label='Other Stars', alpha=0.4, s=30
    )
    sns.scatterplot(
        x='st_met', y='st_mass', data=earth_like,
        color='blue', label='Earth-like Planets', s=80, edgecolor="w", alpha=0.8
    )

    # Highlight solar values
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Solar Metallicity [Fe/H] = 0')
    plt.axhline(1, color='green', linestyle='--', linewidth=1.5, label='Solar Mass [1 Solar Mass]')

    # Annotate only the 2 most extreme outliers per axis (highest/lowest metallicity and highest mass)
    if not outliers.empty:
        extreme_outliers = pd.concat([
            outliers.nsmallest(2, 'st_met'),  # Lowest metallicity
            outliers.nlargest(2, 'st_met'),   # Highest metallicity
            outliers.nlargest(2, 'st_mass')   # Highest mass
        ]).drop_duplicates()

        for _, row in extreme_outliers.iterrows():
            plt.annotate(
                f"({row['st_met']:.2f}, {row['st_mass']:.2f})",
                (row['st_met'], row['st_mass']),
                textcoords="offset points",
                xytext=(10, 10), ha='center',
                fontsize=8, color='darkred',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.7)
            )

    plt.xlabel("Metallicity [Fe/H]", fontsize=14, fontweight="bold")
    plt.ylabel("Stellar Mass [Solar Mass]", fontsize=14, fontweight="bold")
    plt.title("Metallicity vs Stellar Mass for Earth-like Planets and Other Stars", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

# Main function to call analyses
def main():
    # Path to your dataset
    file_path = "../databases/PS_2025.01.12_10.15.56.csv"
    # Indices of Earth-like planets
    earth_like_indices = [596, 708, 1697, 1918, 2406, 2667, 3726, 3843, 5858, 5859, 5862, 5864, 5871, 5873, 5877, 6221, 6224, 6227, 6228, 6332, 6333, 6342, 6346, 6348, 7145, 7148, 7149, 7988, 7989, 7992, 7994, 8253, 8451, 8453, 8454, 8455, 8456, 8723, 8725, 11039, 11041, 11044, 11045, 13653, 13707, 13711, 13712, 13897, 13899, 13901, 13919, 13920, 13922, 13923, 14071, 14144, 14147, 14149, 14164, 14165, 14167, 14178, 14179, 14183, 14184, 14627, 14708, 14712, 14713, 14988, 14990, 15038, 15039, 15048, 15057, 15061, 15068, 15070, 15073, 15078, 15080, 15081, 15089, 15092, 15278, 15347, 15469, 15471, 15472, 15473, 15476, 15724, 15728, 15729, 15730, 15867, 15869, 15870, 16265, 16328, 16329, 16330, 16384, 16385, 16388, 17673, 17676, 17682, 17747, 17748, 17751, 17755, 18929, 18931, 19859, 19861, 19863, 19865, 19866, 23107, 23294, 23298, 23299, 23303, 26302, 26305, 26308, 26564, 26567, 26574, 26600, 26607, 26608, 26612, 26834, 26837, 26838, 27622, 28330, 28332, 28334, 28337, 28339, 29387, 29388, 29392, 29394, 29396, 29397, 30302, 30303, 30304, 30308, 30311, 30544, 30545, 30546, 30548, 32466, 32738, 32740, 32744, 35048, 35055, 35056, 35057, 35167, 35747, 35977, 36078, 37823, 38047, 38048]

    # Load data
    data, earth_like, all_other_stars = load_data(file_path, earth_like_indices)

    # Analyze metallicity29392, 29394, 29396, 29397, 30302, 30303, 30304, 30308, 30311, 30544, 30545, 30546, 30548, 32466, 32738, 32740, 32744, 35048, 35055, 35056, 35057, 35167, 35747, 35977, 36078, 37823, 38047, 38048]
    analyze_metallicity(earth_like, all_other_stars)

    # Analyze stellar mass
    analyze_stellar_mass(earth_like, all_other_stars)

    # Identify and print outliers
    outliers = identify_outliers(earth_like, all_other_stars)

    # Combined analysis with annotations and solar lines
    combined_analysis_with_outliers(earth_like, all_other_stars, outliers) 

# Run the main function
if __name__ == "__main__":
    main()

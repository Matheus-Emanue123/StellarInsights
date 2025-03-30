import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import dcor  # Certifique-se de instalar: pip install dcor
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import os

# Configuração dos diretórios de saída
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
TXT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'txt')
CSV_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'csv')
HTML_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'html')

for directory in [TXT_OUTPUT_DIR, CSV_OUTPUT_DIR, HTML_OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Função: Carregar dataset usando nomes dos planetas ao invés de índices
def load_data(file_path, earth_like_names):
    data = pd.read_csv(file_path, comment='#', low_memory=False)
    
    # Se não existir a coluna "planet_name", mas existir "pl_name", renomeá-la.
    if "planet_name" not in data.columns and "pl_name" in data.columns:
        data.rename(columns={"pl_name": "planet_name"}, inplace=True)
    
    # Filtrar os planetas que estão na lista de nomes
    earth_like = data[data["planet_name"].isin(earth_like_names)]
    
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
        "index": 38048  # Ajuste conforme necessário
    }

    # Criar entradas artificiais para Earth e Similar Earth
    artificial_entries = pd.DataFrame([earth_reference, similar_earth_reference])
    artificial_entries.set_index("index", inplace=True)
    earth_like = pd.concat([earth_like, artificial_entries])
    
    # Selecionar os demais objetos que não são Earth-like (com base na coluna "planet_name")
    all_other_stars = data[~data["planet_name"].isin(earth_like_names)]
    
    return data, earth_like, all_other_stars

# Função: Analisar Metalicidade
def analyze_metallicity(earth_like, all_other_stars):
    sns.set_theme(style="whitegrid", palette="deep")
    
    earth_like_metallicity = earth_like['st_met']
    other_stars_metallicity = all_other_stars['st_met']

    plt.figure(figsize=(12, 6))
    sns.kdeplot(earth_like_metallicity, color='blue', label='Earth-like Planets', fill=True, alpha=0.6, linewidth=2)
    sns.kdeplot(other_stars_metallicity, color='red', label='Other Stars', fill=True, alpha=0.6, linewidth=2)
    plt.xlabel("Metallicity [Fe/H]", fontsize=14, fontweight="bold")
    plt.ylabel("Density", fontsize=14, fontweight="bold")
    plt.title("Comparison of Metallicity between Earth-like Planets and Other Stars", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    stat, p_value = ttest_ind(earth_like_metallicity.dropna(), other_stars_metallicity.dropna())
    print(f"Metallicity - t-statistic: {stat}, p-value: {p_value}")

# Função: Analisar Massa Estelar
def analyze_stellar_mass(earth_like, all_other_stars):
    sns.set_theme(style="whitegrid", palette="muted")
    
    earth_like_mass = earth_like['st_mass']
    other_stars_mass = all_other_stars['st_mass']

    plt.figure(figsize=(12, 6))
    sns.kdeplot(earth_like_mass, color='green', label='Earth-like Planets', fill=True, alpha=0.6, linewidth=2)
    sns.kdeplot(other_stars_mass, color='orange', label='Other Stars', fill=True, alpha=0.6, linewidth=2)
    plt.xlabel("Stellar Mass [Solar Mass]", fontsize=14, fontweight="bold")
    plt.ylabel("Density", fontsize=14, fontweight="bold")
    plt.title("Comparison of Stellar Mass between Earth-like Planets and Other Stars", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    stat, p_value = ttest_ind(earth_like_mass.dropna(), other_stars_mass.dropna())
    print(f"Stellar Mass - t-statistic: {stat}, p-value: {p_value}")

# Função: Identificar Outliers
def identify_outliers(earth_like, all_other_stars):
    outliers = all_other_stars[
        (all_other_stars['st_met'] > 1.5) | 
        (all_other_stars['st_met'] < -1.5) |
        (all_other_stars['st_mass'] > 3)
    ]
    
    print("Outliers identified:")
    if 'planet_name' in outliers.columns:
        print(outliers[['st_met', 'st_mass', 'planet_name']])
    else:
        print(outliers[['st_met', 'st_mass']])
    
    return outliers

# Função: Análise Combinada com Outliers
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

    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Solar Metallicity [Fe/H] = 0')
    plt.axhline(1, color='green', linestyle='--', linewidth=1.5, label='Solar Mass [1 Solar Mass]')

    if not outliers.empty:
        extreme_outliers = pd.concat([
            outliers.nsmallest(2, 'st_met'),
            outliers.nlargest(2, 'st_met'),
            outliers.nlargest(2, 'st_mass')
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

# Função: Análise de Correlação Linear para Earth-like
def analyze_mass_density_correlation(earth_like):
    df = earth_like[['st_mass', 'pl_dens']].dropna()
    corr, p_value = pearsonr(df['st_mass'], df['pl_dens'])
    print(f"Pearson correlation (Earth-like) between stellar mass and planetary density: {corr:.4f} (p-value: {p_value:.4e})")
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x='st_mass', y='pl_dens', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.xlabel("Stellar Mass [Solar Mass]", fontsize=14, fontweight="bold")
    plt.ylabel("Planetary Density [g/cm³]", fontsize=14, fontweight="bold")
    plt.title("Correlation between Stellar Mass and Planetary Density (Earth-like)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Nova Função: Análise de Relação Não Linear para Earth-like
def analyze_nonlinear_relationship(earth_like, output_file):
    df = earth_like[['st_mass', 'pl_dens']].dropna()
    X = df['st_mass'].values.reshape(-1, 1)
    y = df['pl_dens'].values

    results = []
    results.append("Análise de Relação Não Linear entre Massa Estelar e Densidade Planetária (Earth-like):")
    
    # Correlação de Pearson (linear)
    pearson_corr, pearson_p = pearsonr(df['st_mass'], df['pl_dens'])
    results.append(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    
    # Correlação de Spearman
    spearman_corr, spearman_p = spearmanr(df['st_mass'], df['pl_dens'])
    results.append(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    
    # Correlação de Kendall
    kendall_corr, kendall_p = kendalltau(df['st_mass'], df['pl_dens'])
    results.append(f"Kendall correlation: {kendall_corr:.4f} (p-value: {kendall_p:.4e})")
    
    # Regressão Polinomial Grau 2
    poly2 = PolynomialFeatures(degree=2)
    X_poly2 = poly2.fit_transform(X)
    lin_reg2 = LinearRegression().fit(X_poly2, y)
    y_pred2 = lin_reg2.predict(X_poly2)
    r2_poly2 = r2_score(y, y_pred2)
    results.append(f"Regressão Polinomial (grau 2): R² = {r2_poly2:.4f}")
    
    # Regressão Polinomial Grau 3
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X)
    lin_reg3 = LinearRegression().fit(X_poly3, y)
    y_pred3 = lin_reg3.predict(X_poly3)
    r2_poly3 = r2_score(y, y_pred3)
    results.append(f"Regressão Polinomial (grau 3): R² = {r2_poly3:.4f}")
    
    # Suavização LOESS
    loess_result = lowess(y, df['st_mass'], frac=0.3)  # Parâmetro de suavização
    loess_fitted = loess_result[:, 1]
    corr_loess = np.corrcoef(y, loess_fitted)[0, 1]
    results.append(f"LOESS: correlação entre valores observados e ajustados = {corr_loess:.4f}")
    
    # Correlação de distância
    dist_corr = dcor.distance_correlation(df['st_mass'].values, df['pl_dens'].values)
    results.append(f"Correlação de distância: {dist_corr:.4f}")
    
    # Informação Mútua
    mi = mutual_info_regression(X, y, random_state=42)
    results.append(f"Informação Mútua: {mi[0]:.4f}")
    
    # Escrever os resultados no arquivo TXT
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")
    
    for line in results:
        print(line)

# Função principal
def main():
    file_path = r"C:\Users\hecla\OneDrive\Área de Trabalho\CEFET\4 periodo\AEDS II\TRABALHOS\Aplicação de Grafos\databases\PS_2025.01.12_10.15.56.csv"
    # Lista de nomes Earth-like conforme especificado
    earth_like_names = [
        'GJ 3138 d', 'GJ 514 b', 'HATS-59 c', 'HD 109286 b', 'HD 164922 b',
        'HD 191939 g', 'HD 95544 b', 'HIP 54597 b', 'KIC 5437945 b', 'KIC 9663113 b',
        'KOI-1783.02', 'KOI-351 g', 'KOI-351 h', 'Kepler-1040 b', 'Kepler-1097 b',
        'Kepler-111 c', 'Kepler-1126 c', 'Kepler-1143 c', 'Kepler-1318 b',
        'Kepler-1514 b', 'Kepler-1519 b', 'Kepler-1533 b', 'Kepler-1536 b',
        'Kepler-1544 b', 'Kepler-1550 b', 'Kepler-1552 b', 'Kepler-1554 b',
        'Kepler-1593 b', 'Kepler-1600 b', 'Kepler-1625 b', 'Kepler-1630 b',
        'Kepler-1632 b', 'Kepler-1633 b', 'Kepler-1635 b', 'Kepler-1636 b',
        'Kepler-1638 b', 'Kepler-1654 b', 'Kepler-1661 b', 'Kepler-167 e',
        'Kepler-1690 b', 'Kepler-1704 b', 'Kepler-174 d', 'Kepler-1746 b',
        'Kepler-1750 b', 'Kepler-186 f', 'Kepler-1868 b', 'Kepler-1981 b',
        'Kepler-22 b', 'Kepler-309 c', 'Kepler-315 c', 'Kepler-421 b', 'Kepler-439 b',
        'Kepler-441 b', 'Kepler-442 b', 'Kepler-452 b', 'Kepler-511 b',
        'Kepler-553 c', 'Kepler-62 f', 'Kepler-69 c', 'Kepler-712 c', 'Kepler-849 b',
        'Kepler-87 c', 'PH2 b', 'TIC 139270665 c', 'TOI-2180 b', 'TOI-4010 e',
        'TOI-4633 c', 'Wolf 1061 d', 'Earth', 'Similar Earth'
    ]
    
    data, earth_like, all_other_stars = load_data(file_path, earth_like_names)
    
    analyze_metallicity(earth_like, all_other_stars)
    analyze_stellar_mass(earth_like, all_other_stars)
    outliers = identify_outliers(earth_like, all_other_stars)
    combined_analysis_with_outliers(earth_like, all_other_stars, outliers)
    
    # Agora, as correlações serão feitas apenas para os Earth-like
    analyze_mass_density_correlation(earth_like)
    
    nonlinear_output_file = os.path.join(TXT_OUTPUT_DIR, "nonlinear_analysis.txt")
    analyze_nonlinear_relationship(earth_like, nonlinear_output_file)

if __name__ == "__main__":
    main()
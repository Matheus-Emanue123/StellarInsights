# Stellar Insights: Investigating Stellar Composition and Mass as Indicators of Earth-like Exoplanets
 
 ## 📄 Research Paper
 This repository contains the code and data used for this work:  
 [ A Graph Driven Approach to Complex Challenges A Case Study on Multiobjective Stellar and Earth Like Exoplanet Clustering](https://github.com/Matheus-Emanue123/StellarInsights/blob/main/artigo/A_Graph_Driven_Approach_to_Complex_Challenges__A_Case_Study_on_Multiobjective_Stellar_and_Earth_Like_Exoplanet_Clustering.pdf)
 
 ## 📌 Overview
 This repository contains code for analyzing stellar metallicity and mass in relation to Earth-like exoplanets using graph-based methods. It includes:
 - Interactive graph construction
 - Cluster analysis and statistical tests
 - Planetary mass and metallicity analysis
 
 ## ⚙️ Installation and Setup
 ### 1️⃣ Prerequisites
 Ensure you have the following installed:
 - Python (>=3.8)
 - GCC/G++ (>=11) for compiling C/C++ dependencies
 
 ### 2️⃣ Clone the repository
 ```bash
 git clone https://github.com/Matheus-Emanue123/StellarInsights.git
 cd StellarInsights
 ```
 
 ### 3️⃣ Install dependencies
 Create a virtual environment and install required Python packages:
 ```bash
 python -m venv venv
 source venv/bin/activate  # On Windows use `venv\Scripts\activate`
 pip install -r requirements.txt
 ```
 
 ## 🚀 Running the Project
 To execute the main analysis and graph construction, run:
 ```bash
 python generate_graphs_and_analysis.py
 ```
 To run clustering and statistical tests:
 ```bash
 python cluster_analysis_and_statistical_tests.py
 ```
 To analyze metallicity and mass:
 ```bash
 python analyze_planet_metallicity_and_mass.py
 ```
 
 ## 📚 Required Libraries
 The following Python libraries are used:
 - `numpy`
 - `pandas`
 - `matplotlib`
 - `seaborn`
 - `networkx`
 - `pyvis`
 - `scipy`
 - `openpyxl`
 
 Ensure all dependencies are installed before running the scripts.
 
 ## 🔗 Outputs
 - Interactive Graphs (`.html` files)
 - Sub-databases for Gephi (`.csv` files)
 - Statistical Analysis (`.txt` files)
 - Clustered Data Visualizations
 
 Feel free to explore and contribute! 🚀

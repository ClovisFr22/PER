
from pathlib import Path

import pandas as pd
from econml.dml import CausalForestDML

# Chemin du dataset depuis le dossier du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hillstrom.csv"

def main():
    print("Démarrage du projet Causal AI")

    # Charger le dataset
    df = pd.read_csv(DATA_PATH)

    print("Dimensions :", df.shape)
    print("Colonnes :", df.columns.tolist())
    print(df.head())

    print("EconML importé avec succès !")

if __name__ == "__main__":
    main()

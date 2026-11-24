import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hillstrom.csv"
df = pd.read_csv(DATA_PATH)

# Comparaison : e-mail hommes versus contrôle
df = df[
    df["segment"].isin(["Mens E-Mail", "No E-Mail"])
].copy()

df["treatment"] = (
    df["segment"] == "Mens E-Mail"
).astype(int)

# Variables connues avant l'intervention
features = [
    "recency", "history", "mens",
    "womens", "newbie", "zip_code", "channel"
]

X = pd.get_dummies(df[features], drop_first=True)
T = df["treatment"].to_numpy()
Y = df["visit"].astype(float).to_numpy()

X_train, X_test, T_train, T_test, Y_train, Y_test = (
    train_test_split(
        X, T, Y,
        test_size=0.30,
        random_state=42,
        stratify=T
    )
)

# Référence : effet moyen observé sur le test randomisé
ate_test = (
    Y_test[T_test == 1].mean()
    - Y_test[T_test == 0].mean()
)

print("ATE expérimental (test) :", ate_test)

# Taux de visite dans chaque groupe
y_treated = Y_test[T_test == 1]
y_control = Y_test[T_test == 0]

p1 = y_treated.mean()
p0 = y_control.mean()

ate = p1 - p0

# Erreur standard de la différence des moyennes
se = np.sqrt(
    y_treated.var(ddof=1) / len(y_treated)
    + y_control.var(ddof=1) / len(y_control)
)

# Intervalle de confiance de Wald à 95 %
ci_lower = ate - 1.96 * se
ci_upper = ate + 1.96 * se

print(f"Taux traité : {p1:.2%}")
print(f"Taux contrôle : {p0:.2%}")
print(f"ATE : {ate:.2%}")
print(
    f"IC 95 % : [{ci_lower:.2%}, {ci_upper:.2%}]"
)

# Estimation des effets hétérogènes
causal_model = CausalForestDML(
    # Prédiction du résultat Y (visite ou non)
    model_y=RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ),

    # Prédiction du traitement T (e-mail ou non)
    model_t=make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    ),

    discrete_treatment=True,

    # Forêt utilisée pour estimer les effets causaux
    n_estimators=400,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

causal_model.fit(
    Y_train,
    T_train,
    X=X_train
)



# Effet causal conditionnel estimé
uplift_test = causal_model.effect(X_test)

print("CATE moyen estimé :", uplift_test.mean())
print("5 premiers CATE :", uplift_test[:5])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

df = pd.read_csv("data/telco.csv")

possible_binary_cols = ["churn", "retire", "voice", "internet", "forward"]

for col in possible_binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

if df["churn"].dtype == "object":
    df["churn"] = df["churn"].astype(str).str.strip()
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
df["churn"] = df["churn"].astype(int)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

df = df.drop(columns=["ID"], errors="ignore")

categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["tenure", "churn"])

df = df.fillna(df.median(numeric_only=True))

T = "tenure"
E = "churn"
X = df.drop(columns=[T, E]).columns.tolist()

models = {
    "Weibull": WeibullAFTFitter(),
    "LogNormal": LogNormalAFTFitter(),
    "LogLogistic": LogLogisticAFTFitter()
}

fitted = {}

print("\nFITTING MODELS")
for name, model in models.items():
    print("→", name)
    model.fit(df, duration_col=T, event_col=E)
    fitted[name] = model

comparison = pd.DataFrame({
    "Model": list(fitted.keys()),
    "AIC": [fitted[m].AIC_ for m in fitted]
}).sort_values("AIC")

print("\nMODEL COMPARISON")
print(comparison)

best_model_name = comparison.iloc[0]["Model"]
best_model = fitted[best_model_name]

sig_rows = best_model.summary[best_model.summary["p"] < 0.05].index
sig_features = set()

for item in sig_rows:

    if isinstance(item, tuple):
        variable = item[0]
        sig_features.add(variable)
        continue

    if isinstance(item, str):
        if "_mu_" in item:
            sig_features.add(item.replace("_mu_", ""))
            continue

        if "_sigma_" in item:
            sig_features.add(item.replace("_sigma_", ""))
            continue

        sig_features.add(item)
        continue

    sig_features.add(str(item))

sig_features = list(sig_features)

sig_features = [c for c in sig_features if c in df.columns]

df_sig = df[[T, E] + sig_features]

final_model = models[best_model_name].__class__()
final_model.fit(df_sig, duration_col=T, event_col=E)
print("\nFINAL MODEL SUMMARY")
print(final_model.summary)

plt.figure(figsize=(10, 6))
t = np.linspace(0, df[T].max(), 300)
median_row = df[X].median().to_frame().T

for name, model in fitted.items():
    S = model.predict_survival_function(median_row, times=t)
    plt.plot(t, S.iloc[:, 0], label=name)

plt.title("Survival Curves (Weibull, LogNormal, LogLogistic)")
plt.xlabel("Tenure")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("survival_curves.png", dpi=200)
plt.show()

df["monthly_revenue"] = df["income"] * 1000 / 12

months = np.arange(1, 13)
clvs = []

for i, row in df_sig.drop(columns=[T, E]).iterrows():
    S = final_model.predict_survival_function(row.to_frame().T, times=months)
    clv = np.sum(S.values.flatten() * df.loc[i, "monthly_revenue"])
    clvs.append(clv)

df["CLV"] = clvs

df.to_csv("output_with_clv.csv", index=False)
print("\nSaved: output_with_clv.csv")

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


csv_path = "/Users/leandrexhepi/Desktop/GarminPaceProjekt"
csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]


df_list = []
for file in csv_files:
    full_path = os.path.join(csv_path, file)
    try:
        df_tmp = pd.read_csv(full_path)
        df_tmp = df_tmp[df_tmp["Runden"] != "Übersicht"]
        df_list.append(df_tmp)
    except Exception as e:
        print(f"Fehler in Datei {file}: {e}")


df = pd.concat(df_list, ignore_index=True)


def pace_to_float(pace_str):
    try:
        if isinstance(pace_str, str) and ":" in pace_str:
            min, sec = map(int, pace_str.strip().split(":"))
            return min + sec / 60
    except:
        return None
    return None

df["Pace_min_per_km"] = df["Ø Pacemin/km"].apply(pace_to_float)


features = [
    "Ø Herzfrequenzbpm",
    "Maximale Herzfrequenzbpm",
    "Anstieg gesamtm",
    "Abstieg gesamtm",
    "KalorienC",
    "Max. Schrittfrequenz (Laufen)spm",
]


for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.dropna(subset=["Pace_min_per_km"] + features)


print("Übrig nach Bereinigung:", df.shape[0], "Zeilen")



X = df[features]
y = df["Pace_min_per_km"]

print("Anzahl NaNs in Zielwert y:", df["Pace_min_per_km"].isna().sum())
print("Anzahl Zeilen gesamt:", len(df))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = RandomForestRegressor()
model.fit(X_train, y_train)

importances = model.feature_importances_
print("\n📊 Feature Importance:")
for feat, score in zip(features, importances): 
    print(f"{feat:40s}: {score:.3f}")


predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f} min/km")


plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Tatsächlicher Pace (min/km)")
plt.ylabel("Vorhergesagter Pace (min/km)")
plt.title("Modell-Vorhersage vs Realität")
plt.grid(True)
plt.tight_layout()
plt.show()

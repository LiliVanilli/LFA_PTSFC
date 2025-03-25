import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Dateien einlesen
energy_file = "/Users/luisafaust/Desktop/PTFSC_Data/Abgaben/Processed_Energy_All.xlsx"
bike_file = "/Users/luisafaust/Desktop/PTFSC_Data/Abgaben/Bike_All.xlsx"

# Daten laden (erste Sheet wird standardmäßig genommen)
energy_df = pd.read_excel(energy_file)
bike_df = pd.read_excel(bike_file)

# Auswahl, welche der beiden Dateien benutzt werden soll
df = bike_df  # oder bike_df, falls du Bike auswerten willst

def calculate_pinball_loss(actuals, forecasts, quantile):
    """Berechnet den Pinball Loss für ein bestimmtes Quantil."""
    errors = actuals - forecasts
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))

def calculate_crps(actuals, quantile_forecasts, quantile_levels):
    """Berechnet den CRPS-Wert basierend auf den gegebenen Quantilen."""
    crps_values = []

    for i in range(len(actuals)):
        sorted_preds = quantile_forecasts[i]
        actual = actuals[i]

        crps = 0
        for j in range(len(quantile_levels) - 1):
            width = quantile_levels[j + 1] - quantile_levels[j]
            left_pred = sorted_preds[j]
            right_pred = sorted_preds[j + 1]

            if actual < left_pred:
                crps += width * (right_pred - left_pred)
            elif actual > right_pred:
                crps += width * (right_pred - left_pred)
            else:
                height = right_pred - left_pred
                prop = (actual - left_pred) / height if height > 0 else 0
                crps += width * height * (1 - prop)

        crps_values.append(crps)

    return np.mean(crps_values)

# Sicherstellen, dass forecast_date als Datum erkannt wird
df["forecast_date"] = pd.to_datetime(df["forecast_date"], format="%d.%m.%y")

# CRPS und Pinball Loss berechnen pro Forecast Date
results = []

for date in df["forecast_date"].unique():
    df_date = df[df["forecast_date"] == date]

    # Pinball Loss pro Quantil
    pinball_losses = {}
    for q in ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]:
        quant = float(q.replace("q", ""))  # z.B. "q0.025" -> 0.025
        pinball_losses[q] = calculate_pinball_loss(
            df_date["Actual"],
            df_date[q],
            quant
        )
    mean_pinball_loss = np.mean(list(pinball_losses.values()))

    # CRPS berechnen (wird nur in DataFrame gespeichert, aber nicht geplottet)
    crps = calculate_crps(
        df_date["Actual"].values,
        df_date[["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]].values,
        np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    )

    results.append({
        "Forecast Date": date,
        "CRPS": crps,  # CRPS wird weiterhin berechnet, aber nicht mehr geplottet
        "Mean Pinball Loss": mean_pinball_loss,
        **pinball_losses
    })

# Ergebnisse als DataFrame
forecast_evaluation_df = pd.DataFrame(results)

# Optional: Spalte "Week" einfügen (laufende Nummer)
forecast_evaluation_df.insert(1, "Week", range(1, len(forecast_evaluation_df) + 1))

print("Ergebnisse (mit CRPS, aber CRPS wird nicht geplottet):")
print(forecast_evaluation_df)

# KIT Farben definieren
KIT_COLORS = {
    'KIT_GRUEN': (0/255, 150/255, 130/255, 1.0),      # Primary
    'KIT_BLAU': (70/255, 100/255, 170/255, 1.0),      # Primary
    'SCHWARZ': (140/255, 182/255, 60/255, 1.0),       # Primary
    'BRAUN': (167/255, 130/255, 46/255, 1.0),         # Accent
    'LILA': (163/255, 16/255, 124/255, 1.0),          # Accent
    'CYAN': (35/255, 161/255, 224/255, 1.0),          # Accent
    'GRAU': (64/255, 64/255, 64/255, 1.0),            # Accent
    'ROT': (162/255, 34/255, 35/255, 1.0)             # Accent
}

# Hellere Farben für die Quantile
quantile_colors = {
    "q0.025": (KIT_COLORS['ROT'][0], KIT_COLORS['ROT'][1], KIT_COLORS['ROT'][2], 0.5),
    "q0.25": (KIT_COLORS['CYAN'][0], KIT_COLORS['CYAN'][1], KIT_COLORS['CYAN'][2], 0.5),
    "q0.5": (KIT_COLORS['BRAUN'][0], KIT_COLORS['BRAUN'][1], KIT_COLORS['BRAUN'][2], 0.5),
    "q0.75": (KIT_COLORS['LILA'][0], KIT_COLORS['LILA'][1], KIT_COLORS['LILA'][2], 0.5),
    "q0.975": (KIT_COLORS['SCHWARZ'][0], KIT_COLORS['SCHWARZ'][1], KIT_COLORS['SCHWARZ'][2], 0.5)
}

# ============ PLOT NUR PINBALL LOSS (kein CRPS) ============

fig, ax = plt.subplots(figsize=(12, 6))

# X-Achse: "Week"
x_values = forecast_evaluation_df["Week"]

# 1) Plot Mean Pinball Loss (durchgezogene Linie, z. B. GRÜN)
ax.plot(x_values,
        forecast_evaluation_df["Mean Pinball Loss"],
        marker='s', linestyle='-', color=KIT_COLORS['KIT_GRUEN'],
        linewidth=2, label="Mean Pinball Loss")

# 2) Plot Pinball Loss pro Quantil (dünnere Linien in verschiedenen Farben)
for quantile, color in quantile_colors.items():
    ax.plot(x_values,
            forecast_evaluation_df[quantile],
            marker='o', linestyle='--', color=color,
            linewidth=1.5, label=f"Pinball Loss {quantile}")

ax.set_xlabel('Forecast Week')
ax.set_ylabel('Pinball Loss')
# ax.set_title('Pinball Loss Entwicklung pro Quantil über die Forecast-Wochen')
ax.grid(True)
ax.legend()

fig.tight_layout()

# Speichern des Plots
plot_dir = "/Users/luisafaust/Desktop/PTFSC_Data/Abgaben/plots"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "pinball_loss_per_quantile.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot gespeichert unter: {plot_path}")
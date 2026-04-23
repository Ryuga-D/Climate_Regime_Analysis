import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Raw data dictionary (parsed from your message)
model_results = {
    "BiLSTM": [
        ["Temp_Anomaly_°C", 1.3678, -8.7331, "93.28%"],
        ["CO2_in_ppm", 1.5605, -33.5292, "96.27%"],
        ["Precip_in_mm", 0.4834, -0.0754, "98.51%"],
        ["Nino34_in_°C", 0.7976, -0.0000, "3.73%"],
        ["Volcanic_Global", 0.5388, -72.9124, "55.97%"],
        ["TSI_Wm2", 0.7988, -0.3774, "61.19%"],
    ],
    "CNN_LSTM": [
        ["Temp_Anomaly_°C", 1.0781, -5.3329, "92.54%"],
        ["CO2_in_ppm", 0.7131, -6.7084, "96.27%"],
        ["Precip_in_mm", 0.4655, 0.0001, "98.51%"],
        ["Nino34_in_°C", 0.5198, 0.5601, "96.27%"],
        ["Volcanic_Global", 0.5357, -73.8390, "42.54%"],
        ["TSI_Wm2", 0.9416, -1.1214, "96.27%"],
    ],
    "Seq2Seq": [
        ["Temp_Anomaly_°C", 1.1913, -6.6236, "92.54%"],
        ["CO2_in_ppm", 1.1493, -18.0735, "96.27%"],
        ["Precip_in_mm", 0.4911, -0.2201, "97.76%"],
        ["Nino34_in_°C", 0.7969, 0.0000, "21.64%"],
        ["Volcanic_Global", 0.5459, -74.8584, "55.97%"],
        ["TSI_Wm2", 0.7803, -0.2634, "13.43%"],
    ],
    "SARIMA": [
        ["Temp_Anomaly_°C", 0.2530, -0.0748, "40.00%"],
        ["CO2_in_ppm", 0.0119, 0.5869, "60.00%"],
        ["Precip_in_mm", 0.3852, 0.5073, "100.00%"],
        ["Nino34_in_°C", 0.4117, -0.0025, "60.00%"],
        ["Volcanic_Global", 0.1639, -1.77e+49, "0.00%"],
        ["TSI_Wm2", 0.4893, -0.0352, "100.00%"],
    ],
    "Ensemble": [
        ["Temp_Anomaly_°C", 1.2151, -6.8559, "93.28%"],
        ["CO2_in_ppm", 1.1211, -17.1424, "95.52%"],
        ["Precip_in_mm", 0.4643, -0.0389, "98.51%"],
        ["Nino34_in_°C", 0.4916, 0.6179, "98.51%"],
        ["Volcanic_Global", 0.5405, -73.6514, "42.54%"],
        ["TSI_Wm2", 0.9139, -0.9844, "97.01%"],
    ]
}

# Start plotting
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Define table structure
col_labels = ["Variable", "MAE", "R²", "Trend Accuracy"]
table_data = []

# Build table row-wise with section headers
for model, rows in model_results.items():
    table_data.append([f"Model: {model}", "", "", ""])
    table_data.extend(rows)

# Build table
table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    colColours=["#40466e"] * len(col_labels),
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

# Style
table.auto_set_font_size(False)
table.set_fontsize(12)

# Format rows
# Format rows (safe from IndexError)
for (row_idx, col_idx), cell in table.get_celld().items():
    if row_idx == 0:
        # Header row
        cell.set_text_props(weight='bold', color='white')
    elif row_idx < len(table_data):
        if table_data[row_idx][0].startswith("Model:"):
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor("#c5d2e2")
        elif col_idx == 0:
            cell.set_text_props(weight='bold')


plt.title("Forecast Model Results Summary", fontsize=16, weight='bold')
plt.savefig('forecast_model_results_summary.png', bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()

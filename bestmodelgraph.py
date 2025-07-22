import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Copy your latest metrics_dict here manually
metrics_dict = {
    "Logistic Regression": {"accuracy": 0.8063, "roc_auc": 0.9048},
    "Decision Tree": {"accuracy": 0.8154, "roc_auc": 0.7478},
    "Random Forest": {"accuracy": 0.85, "roc_auc": 0.91}  # Example
}

model_names = list(metrics_dict.keys())
accuracies = [metrics_dict[name]["accuracy"] for name in model_names]
roc_aucs = [metrics_dict[name]["roc_auc"] for name in model_names]

metrics_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies,
    "ROC-AUC": roc_aucs
})

metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_long, x="Model", y="Score", hue="Metric")
plt.title("🔍 Model Performance Comparison (Accuracy & ROC-AUC)")
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("model/model_performance_comparison.png")
plt.show()

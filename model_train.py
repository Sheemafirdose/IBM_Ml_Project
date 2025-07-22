import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

# -------------------------
# Step 1: Load Dataset
# -------------------------
df = pd.read_csv("data/income.csv")
print(" Data loaded. Shape:", df.shape)
print("dataset:\n", df.head(10))

# -------------------------
# Step 2: Data Cleaning & Feature Selection
# -------------------------
df.replace("?", np.nan, inplace=True)

# Drop less useful or redundant columns
df.drop(columns=["education", "fnlwgt", "race", "relationship"], inplace=True)

# Keep only useful workclasses
valid_workclass = [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov"
]
df = df[df["workclass"].isin(valid_workclass)]

# Filter out unrealistic ages
df = df[(df["age"] >= 17) & (df["age"] <= 75)]

# Drop rows with missing values
df.dropna(inplace=True)

# Encode income
df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

print(" Cleaned shape:", df.shape)
print("dataset:\n", df.head(10))

# -------------------------
# Step 3: Visualization
# -------------------------
sns.countplot(data=df, x='income')
plt.title("Income Class Distribution")
plt.savefig("model/income_distribution.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="age", hue="income", bins=30, kde=True)
plt.title("Age vs Income")
plt.savefig("model/age_income_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("model/numeric_correlation.png")
plt.close()

# -------------------------
# Step 4: Split Data
# -------------------------
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -------------------------
# Step 5: Upsample for Class Balance
# -------------------------
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data.income == 0]
minority = train_data[train_data.income == 1]

minority_upsampled = resample(
    minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE
)
train_balanced = pd.concat([majority, minority_upsampled])

X_train = train_balanced.drop("income", axis=1)
y_train = train_balanced["income"]

# -------------------------
# Step 6: Preprocessing
# -------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# -------------------------
# Step 7: Define Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=1.5,
        random_state=RANDOM_STATE
    )
}

metrics_dict = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    roc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    acc = accuracy_score(y_test, y_pred)

    print(f"\n {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred))

    metrics_dict[name] = {"accuracy": acc, "roc_auc": roc}

    disp = ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, display_labels=["<=50K", ">50K"],
        cmap="Blues", normalize="true"
    )
    disp.ax_.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"model/{name.lower().replace(' ', '_')}_cm.png")
    plt.close()

# -------------------------
# Step 8: Save Best Model
# -------------------------
best_model_name = max(metrics_dict, key=lambda x: metrics_dict[x]["roc_auc"])
print(f"\n Best model (by ROC-AUC): {best_model_name}")

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name])
])
best_model.fit(X, y)

with open("model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(" Model saved at: model/best_model.pkl")

# -------------------------
# Step 9: Predict on Test Set
# -------------------------
y_pred = best_model.predict(X_test)
predictions_df = X_test.copy()
predictions_df["Actual"] = y_test.values
predictions_df["Predicted"] = y_pred
print("\n🔮 Sample Predictions (Actual vs Predicted):")
print(predictions_df.head(10))
predictions_df.to_csv("model/predictions_on_test.csv", index=False)
print(" Saved predictions as: model/predictions_on_test.csv")

# -------------------------
# Step 10: Final Evaluation
# -------------------------
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print("\n Final Model Evaluation:")
print(f"🔹 Accuracy     : {acc:.4f}")
print(f"🔹 ROC-AUC      : {roc:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["<=50K", ">50K"],
    cmap="Blues", normalize="true"
)
disp.ax_.set_title("Final Confusion Matrix - Best Model")
plt.tight_layout()
plt.savefig("model/final_model_confusion_matrix.png")
plt.show()

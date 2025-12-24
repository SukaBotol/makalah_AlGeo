import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("features.csv")
feature_cols = [col for col in df.columns if col.startswith("feature_")]
X = df[feature_cols].values
y_agent = df["agent"].values
y_material = df["material"].values



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = 20
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA:")
print(f"  Original dimensions: 40")
print(f"  After PCA: {n_components}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

print(f"  Top 5 components explain: {pca.explained_variance_ratio_[:5]}")

indices = np.arange(len(X_pca))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_pca = X_pca[train_idx]
X_test_pca = X_pca[test_idx]
y_agent_train = y_agent[train_idx]
y_agent_test = y_agent[test_idx]
y_material_train = y_material[train_idx]
y_material_test = y_material[test_idx]

print(f"\nTrain-test split:")
print(f"  Train: {len(train_idx)} samples")
print(f"  Test: {len(test_idx)} samples")

print("\n" + "="*50)
print("Training AGENT classifier...")
print("="*50)

agent_classifier = SVC(kernel='linear', C=1.0, random_state=42)
agent_classifier.fit(X_train_pca, y_agent_train)

# Evaluate
y_agent_pred = agent_classifier.predict(X_test_pca)
agent_acc = accuracy_score(y_agent_test, y_agent_pred)

print(f"\nAgent classifier accuracy: {agent_acc:.2%}")
print("\nConfusion matrix (agent):")
print(confusion_matrix(y_agent_test, y_agent_pred))
print("\nClassification report (agent):")
print(classification_report(y_agent_test, y_agent_pred))

print("\n" + "="*50)
print("Training MATERIAL classifier...")
print("="*50)

material_classifier = SVC(kernel='linear', C=1.0, random_state=42)
material_classifier.fit(X_train_pca, y_material_train)

# Evaluate
y_material_pred = material_classifier.predict(X_test_pca)
material_acc = accuracy_score(y_material_test, y_material_pred)

print(f"\nMaterial classifier accuracy: {material_acc:.2%}")
print("\nConfusion matrix (material):")
print(confusion_matrix(y_material_test, y_material_pred))
print("\nClassification report (material):")
print(classification_report(y_material_test, y_material_pred))

joblib.dump(scaler, "scaler.joblib")
joblib.dump(pca, "pca.joblib")
joblib.dump(agent_classifier, "agent_model.joblib")
joblib.dump(material_classifier, "material_model.joblib")

print("\n" + "="*50)
print("Models saved!")
print("  scaler.joblib")
print("  pca.joblib")
print("  agent_model.joblib")
print("  material_model.joblib")
print("="*50)
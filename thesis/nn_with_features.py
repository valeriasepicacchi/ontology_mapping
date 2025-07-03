import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import ace_tools as tools
import matplotlib.pyplot as plt

''' These experiments for a neural network were run on a company owned dataset and are therefore not reproducible'''
# Load the dataset with extracted features
df_features = pd.read_csv("taxonomy_concept_features.csv")
concept_dict = {row["Concept"]: row.drop(["Concept", "Preferred Label"]).values for _, row in df_features.iterrows()}

# Generate Positive (Equivalent) and Negative (Non-Equivalent) Pairs
positive_pairs = []
negative_pairs = []

# Randomly create equivalent (positive) pairs based on existing concepts
for concept in df_features["Concept"]:
    for other_concept in df_features["Concept"]:
        if concept != other_concept:
            # Use LexicalMatcher or DirectXRefMatcher to determine equivalent pairs
            if df_features[df_features["Concept"] == concept]["LexicalMatcher"].values[0] == 1:
                positive_pairs.append((concept, other_concept, 1))
            else:
                negative_pairs.append((concept, other_concept, 0))

# Ensure balanced dataset
negative_pairs = random.sample(negative_pairs, min(len(negative_pairs), len(positive_pairs)))
pairs = positive_pairs + negative_pairs
random.shuffle(pairs)
X = []
y = []

for c1, c2, label in pairs:
    if c1 in concept_dict and c2 in concept_dict:
        X.append(np.concatenate((concept_dict[c1], concept_dict[c2])))  # Concatenate features
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define Neural Network Model
class ConceptClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ConceptClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize Model
input_dim = X_train.shape[1]
model = ConceptClassifier(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate Model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().numpy()
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc_score:.4f}")

# Feature Importance using SHAP
explainer = shap.Explainer(model, X_train_tensor)
shap_values = explainer(X_test_tensor)
feature_importance = np.abs(shap_values.values).mean(axis=0)
feature_names = list(df_features.drop(["Concept", "Preferred Label"], axis=1).columns)
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
tools.display_dataframe_to_user(name="Feature Importance", dataframe=feature_importance_df)
print("Feature importance computed and displayed.")


plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance for Concept Equivalence Classification")
plt.gca().invert_yaxis()
plt.show()

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load data
df = pd.read_csv('../docs/Crop_recommendation.csv')

# Define features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# SVC
svc = SVC(kernel='rbf')  # You can try different kernels here
svc.fit(X_train_scaled, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(max_depth=9, n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Gradient Boosting
grad = GradientBoostingClassifier()
grad.fit(X_train, y_train)

# LightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train, y_train)

# Voting Classifier
final_model = VotingClassifier(estimators=[('rf', rf), ('grad', grad), ('lgbm', lgbm)], voting='hard')
final_model.fit(X_train, y_train)

# Evaluation

models = {
    'KNN': knn,
    'SVC': svc,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': grad,
    'LightGBM': lgbm,
    'Voting Classifier': final_model
}

for name, model in models.items():
    print(f"Evaluating {name}...")
    print("Accuracy:", model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")

# Check if the "models" directory exists, create it if not
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save models
for name, model in models.items():
    joblib.dump(model, f'{models_dir}/{name.lower().replace(" ", "_")}_model.joblib')

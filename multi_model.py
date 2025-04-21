import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                             VotingClassifier, 
                             StackingClassifier)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv(r'C:\MALICIOUS-URL-DETECTION\Dataset\Dataset.csv')

# Data Quality Checks
print("Initial Data Overview:")
print(f"Total samples: {len(df)}")
print(f"Class Distribution:\n{df['target'].value_counts(normalize=True)}")
print(f"Missing values:\n{df.isnull().sum()}")

# Clean data
df = df.dropna().reset_index(drop=True)
df = df.drop_duplicates()

# Feature-Target Separation
X = df.drop(columns=['target'])
y = df['target']

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Initialize Models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight_dict,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        class_weight=class_weight_dict,
        max_depth=8,
        min_samples_split=10,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        scale_pos_weight=class_weights[1]/class_weights[0],
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    'Voting Ensemble': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200,
                class_weight=class_weight_dict,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42)),
            ('xgb', XGBClassifier(
                scale_pos_weight=class_weights[1]/class_weights[0],
                use_label_encoder=False,
                eval_metric='logloss')),
            ('dt', DecisionTreeClassifier(
                class_weight=class_weight_dict,
                max_depth=8,
                min_samples_split=10,
                random_state=42))
        ],
        voting='soft'
    ),
    'Stacking Ensemble': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200,
                class_weight=class_weight_dict,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42)),
            ('xgb', XGBClassifier(
                scale_pos_weight=class_weights[1]/class_weights[0],
                use_label_encoder=False,
                eval_metric='logloss')),
            ('dt', DecisionTreeClassifier(
                class_weight=class_weight_dict,
                max_depth=8,
                min_samples_split=10,
                random_state=42))
        ],
        final_estimator=LogisticRegression()
    )
}

# Training and Evaluation
results = {}
print("\n=== Model Training and Evaluation ===")
for name, model in models.items():
    print(f"\n-- {name} --")
    
    # Cross-validation
    if name not in ['Voting Ensemble', 'Stacking Ensemble']:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Print metrics
    print(f"Accuracy: {results[name]['accuracy']:.2f}")
    print(f"Precision: {results[name]['precision']:.2f}")
    print(f"Recall: {results[name]['recall']:.2f}")
    print(f"F1-Score: {results[name]['f1']:.2f}")
    
    # Print confusion matrix for the best model
    if name == 'Stacking Ensemble':
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Feature Importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Features:")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        print(importances.sort_values(ascending=False).head(5))

print("\n=== Model Performance Summary ===")
performance_df = pd.DataFrame(results).T
print(performance_df)

# Save the best model (based on F1-score)
best_model_name = performance_df['f1'].idxmax()
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name} (F1-Score: {performance_df.loc[best_model_name, 'f1']:.2f})")

# Save ensemble models
ensemble_data = {
    'voting_model': models['Voting Ensemble'],
    'stacking_model': models['Stacking Ensemble'],
    'best_model': best_model,
    'best_model_name': best_model_name,
    'feature_names': list(X.columns),
    'class_names': ['Legitimate', 'Phishing']
}

with open("phishing_ensemble_models.pkl", "wb") as f:
    pickle.dump(ensemble_data, f)

print("\nModels saved as phishing_ensemble_models.pkl")

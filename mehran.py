import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import numpy as np
# Assuming dataset_path is already set and points to a CSV file.
dataset_path = r"C:\Users\pooyaPc\Desktop\OVS.csv"
df = pd.read_csv(dataset_path)

# Data preprocessing as per your original specification
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
df.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1, inplace=True)

X = df.drop('Label', axis=1)
y = df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_scaled_imputed = imputer.fit_transform(X_scaled)

smote = SMOTE()
pca = PCA(n_components=0.95)

rf_classifier = RandomForestClassifier(n_estimators=100)
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('xgb', xgb_classifier)
], voting='soft')

pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('smote', smote),
    ('pca', pca),
    ('classifier', voting_clf)
])

X_train, X_test, y_train, y_test = train_test_split(X_scaled_imputed, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
model_filename = 'trained_model.pkl'
joblib.dump(pipeline, model_filename)
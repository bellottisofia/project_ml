import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from shapely.geometry import shape
from shapely import wkt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# Mapping for the classes
change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5}

# Load CSV data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Feature Engineering - Geometric Features
def calculate_area(geometry):
    geom = wkt.loads(geometry)
    return geom.area


def calculate_perimeter(geometry):
    geom = wkt.loads(geometry)
    return geom.length


def calculate_bounding_box(geometry):
    geom = wkt.loads(geometry)
    minx, miny, maxx, maxy = geom.bounds
    return maxx - minx, maxy - miny  # Return width and height of bounding box


def calculate_compactness(geometry):
    geom = wkt.loads(geometry)
    area = geom.area
    perimeter = geom.length
    return area ** 2 / perimeter ** 2 if perimeter > 0 else 0


# Calculate features if geometry exists
if 'geometry' in train_df.columns:
    train_df['area'] = train_df['geometry'].apply(calculate_area)
    train_df['perimeter'] = train_df['geometry'].apply(calculate_perimeter)
    train_df['bbox_width'], train_df['bbox_height'] = zip(*train_df['geometry'].apply(calculate_bounding_box))
    train_df['compactness'] = train_df['geometry'].apply(calculate_compactness)

    test_df['area'] = test_df['geometry'].apply(calculate_area)
    test_df['perimeter'] = test_df['geometry'].apply(calculate_perimeter)
    test_df['bbox_width'], test_df['bbox_height'] = zip(*test_df['geometry'].apply(calculate_bounding_box))
    test_df['compactness'] = test_df['geometry'].apply(calculate_compactness)

# Temporal Features (if date columns exist)
if 'date0' in train_df.columns and 'change_status_date0' in train_df.columns:
    # Calculate time difference between consecutive date columns and change_status_date columns
    for i in range(4):  # for date0 to date4
        # Parse dates with the correct format (dayfirst=True)
        train_df[f'days_between_{i}'] = (pd.to_datetime(train_df[f'date{i+1}'], dayfirst=True) - pd.to_datetime(train_df[f'date{i}'], dayfirst=True)).dt.days
        test_df[f'days_between_{i}'] = (pd.to_datetime(test_df[f'date{i+1}'], dayfirst=True) - pd.to_datetime(test_df[f'date{i}'], dayfirst=True)).dt.days


# One-Hot Encoding for Categorical Features
categorical_features = ['urban_type', 'geography_type']
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(train_df[categorical_features].fillna('Unknown')).toarray()
encoded_test_features = encoder.transform(test_df[categorical_features].fillna('Unknown')).toarray()

# Numerical Features from Satellite Image Data
numerical_features = [col for col in train_df.columns if 'img_' in col]

# Create the final feature dataset
X = np.hstack((train_df[numerical_features].values, encoded_features,
               train_df[['area', 'perimeter', 'bbox_width', 'bbox_height', 'compactness']].values,
               train_df[[f'days_between_{i}' for i in range(4)]].values))  # Include temporal differences
y = train_df['change_type'].map(change_type_map).values
X_test = np.hstack((test_df[numerical_features].values, encoded_test_features,
                    test_df[['area', 'perimeter', 'bbox_width', 'bbox_height', 'compactness']].values,
                    test_df[[f'days_between_{i}' for i in range(4)]].values))

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can choose different strategies like 'median' or 'most_frequent'
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Optional: Apply PCA for Dimensionality Reduction
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

# Now, you can use X_pca and X_test_pca in your machine learning models
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train and evaluate models

"""# 1. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
print("Random Forest Classifier Report:")
print(classification_report(y_val, rf_pred))

# 2. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_val)
print("SVM Classifier Report:")
print(classification_report(y_val, svm_pred))

# 3. Gradient Boosting Machine (GBM)
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)
gbm_pred = gbm_model.predict(X_val)
print("Gradient Boosting Machine Classifier Report:")
print(classification_report(y_val, gbm_pred))

# 4. Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_val)
print("Neural Network Classifier Report:")
print(classification_report(y_val, nn_pred))
"""
# 1. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)

# Print classification report
print("Random Forest Classifier Report:")
print(classification_report(y_val, rf_pred))

# Calculate Mean F1-Score
rf_f1_score = f1_score(y_val, rf_pred, average='macro')
print(f"Random Forest Mean F1-Score: {rf_f1_score:.4f}")

# 2. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_val)

# Print classification report
print("SVM Classifier Report:")
print(classification_report(y_val, svm_pred))

# Calculate Mean F1-Score
svm_f1_score = f1_score(y_val, svm_pred, average='macro')
print(f"SVM Mean F1-Score: {svm_f1_score:.4f}")

# 3. Gradient Boosting Machine (GBM)
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)
gbm_pred = gbm_model.predict(X_val)

# Print classification report
print("Gradient Boosting Classifier Report:")
print(classification_report(y_val, gbm_pred))

# Calculate Mean F1-Score
gbm_f1_score = f1_score(y_val, gbm_pred, average='macro')
print(f"GBM Mean F1-Score: {gbm_f1_score:.4f}")

# 4. Neural Network (MLP)
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_val)

# Print classification report
print("Neural Network Classifier Report:")
print(classification_report(y_val, nn_pred))

# Calculate Mean F1-Score
nn_f1_score = f1_score(y_val, nn_pred, average='macro')
print(f"Neural Network Mean F1-Score: {nn_f1_score:.4f}")

# Optionally, you can also compute the overall Mean F1-Score for all models
mean_f1_scores = [rf_f1_score, svm_f1_score, gbm_f1_score, nn_f1_score]
best_model_index = np.argmax(mean_f1_scores)

print(f"\nBest Model Based on Mean F1-Score: Model {best_model_index + 1}")
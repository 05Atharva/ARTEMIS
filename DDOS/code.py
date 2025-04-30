#Install the Xgboost model
pip install xgboost

#Import the required libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import glob

# Path to the dataset folder (Modify accordingly)
data_path = "C:/Users/Lenovo/DDOS Model/archive (4)"

# Load all Parquet files
def load_parquet_files(folder):
    all_files = glob.glob(os.path.join(folder, "*.parquet"))
    df_list = [pd.read_parquet(file) for file in all_files]
    return pd.concat(df_list, ignore_index=True)

# Load dataset
df = load_parquet_files(data_path)

# Drop null values and duplicate rows
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical labels
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Separate features and labels
X = df.drop(columns=['Label'])
y = df['Label']

# Handle data imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# XGBClassifier
# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=200,
#               n_jobs=None, num_parallel_tree=None, ...)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy: 0.7524311952852321
# Confusion Matrix:
#  [[24363    13     0     0     2     0     0     0     0     0     1     5
#       1     0     1     0     0     3]
#  [    1 12556  2496  2198    23   488  2073    58  3380   538   246   303
#       0     0    30    24    34     4]
#  [    0  1074 12896    52     0    16  2855     0  7274    23     0     3
#       0     0     0     2     0     0]
#  [    0   508   195 14470     2  1385   648    21   288  6418     0    52
#       1     4     4     7   175     0]
#  [   12    16     1     8 24004     4     1     1     0     4     5     7
#       5     1     0    11     2    33]
#  [    1   260    45  1651     0 10264   148    50   114   402  9039  2452
#       0     1    27     3    75     0]
#  [    0   671  2117    49     2   243 19065     2  1523    65   400    74
#       0     0     1     1     1     2]
#  [    0    95     6   265     0   145     4 16633     2    71    29    37
#       1     0  6471   451    75     0]
#  [    0  1205  5189   113     0    39  2710     0 15020    52     0     1
#       0     0     0     0     1     0]
#  [    1   411   117  9670     1  1163   619    21   222 11537     1    90
#       1     5     8     6   251     0]
#  [    0   100     0    21     1  1881     6     7     0     6 19608  2642
#       0     0     1     2     0     0]
#  [    0   148     0   102     0   937     0    10    14    85  5556 17358
#       1     1    17     4   202     0]
#  [    5     0     0     1     0     2     0     0     2     2     0    80
#   23986     0     0   138    32     0]
#  [    4     1     0     6     3     0     0     0     1     2     0     3
#       3 24232     0    51     2     6]
#  [    0    71     0   230     0   122     0  8884     0   143    42    23
#       0     0 14036   552    93     0]
#  [    1    53    21    32     0     9    41  1872    17    17     0    19
#       0     0  1640 20302    84     0]
#  [    0     0     0    66     0     0     0     1     0    19     0    18
#       0     0     0     0 24203     0]
#  [    0     0     0     0     3     0     0     0     0     0     0     0
#       0     0     0     0     0 24223]]

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00     24389
#            1       0.73      0.51      0.60     24452
#            2       0.56      0.53      0.55     24195
#            3       0.50      0.60      0.54     24178
#            4       1.00      1.00      1.00     24115
#            5       0.61      0.42      0.50     24532
#            6       0.68      0.79      0.73     24216
#            7       0.60      0.68      0.64     24285
#            8       0.54      0.62      0.58     24330
#            9       0.60      0.48      0.53     24124
#           10       0.56      0.81      0.66     24275
#           11       0.75      0.71      0.73     24435
#           12       1.00      0.99      0.99     24248
#           13       1.00      1.00      1.00     24314
#           14       0.63      0.58      0.60     24196
#           15       0.94      0.84      0.89     24108
#           16       0.96      1.00      0.98     24307
#           17       1.00      1.00      1.00     24226

#     accuracy                           0.75    436925
#    macro avg       0.76      0.75      0.75    436925
# weighted avg       0.76      0.75      0.75    436925


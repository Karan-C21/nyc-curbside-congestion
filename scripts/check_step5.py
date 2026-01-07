import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv('data/modeling_dataset.csv')
print(f'Dataset: {len(df)} rows')
print(f'\nTarget distribution:')
print(df['high_congestion'].value_counts())

X = df[['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'month']]
y = df['high_congestion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f'\nStep 5 Validation:')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')

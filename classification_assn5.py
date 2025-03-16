from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

train_data = pd.read_csv('crop_train.csv')
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']

test_data = pd.read_csv('crop_test.csv')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
test_data = scaler.transform(test_data)

rf_clf = RandomForestClassifier(n_estimators=100)

rf_clf.fit(X_train, y_train)

predictions = rf_clf.predict(test_data)

submission_ids = range(len(predictions))

submission = pd.DataFrame({
    'ID': submission_ids,
    'Class': predictions
})

submission.to_csv('submission4.csv', index=False)
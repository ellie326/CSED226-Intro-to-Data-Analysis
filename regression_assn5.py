import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

train = pd.read_csv('age_train.csv')
test = pd.read_csv('age_test.csv')

train = pd.get_dummies(train, columns=['gender'])
test = pd.get_dummies(test, columns=['gender'])

y_train = train['age']
X_train = train.drop('age', axis=1)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
test = scaler.transform(test)

rf = RandomForestRegressor(random_state=20)
rf.fit(X_train, y_train)

pred_train = rf.predict(X_train)

pred_test = rf.predict(test)

submission_ids = range(0, len(pred_test))

submission = pd.DataFrame({
    'ID': submission_ids,
    'age': pred_test
})

submission.to_csv('submission11.csv', index=False)
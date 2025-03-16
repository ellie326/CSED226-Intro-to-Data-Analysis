import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor

# 데이터 불러오기
train_data = pd.read_csv('age_train.csv')
test_data = pd.read_csv('age_test.csv')

# 'gender' 열 원-핫 인코딩
train_data = pd.get_dummies(train_data, columns=['gender'])
test_data = pd.get_dummies(test_data, columns=['gender'])

# 레이블과 피쳐 분리
X_train = train_data.drop('age', axis=1)
y_train = train_data['age']
X_test = test_data

# 결측치 처리: 평균값으로 채우기
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# 불리언 타입 데이터를 숫자로 변환
X_train = X_train*1
X_test = X_test*1

# 이상치 제거: IQR을 사용해 이상치 탐지 후 제거
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
filtered_entries = ((X_train >= (Q1 - 1.5 * IQR)) & (X_train <= (Q3 + 1.5 * IQR))).all(axis=1)
X_train = X_train[filtered_entries]
y_train = y_train[filtered_entries]

# 데이터 전처리: 로버스트 정규화 적용
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 학습
knn = KNeighborsRegressor(n_neighbors=15)
knn.fit(X_train, y_train)

# 예측
predictions = knn.predict(X_test)

# 예측 결과를 데이터프레임으로 변환
submission = pd.DataFrame()
submission['ID'] = test_data.index
submission['age'] = predictions

# 결과를 csv 파일로 저장
submission.to_csv('submission13.csv', index=False)
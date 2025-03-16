import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

# 데이터 로드
train_data = pd.read_csv('crop_train.csv')
test_data = pd.read_csv('crop_test.csv')

# 결측치 제거
train_data = train_data.dropna()

# 학습 데이터와 레이블 분리
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data

# RobustScaler를 사용한 정규화 적용
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# k-Nearest Neighbor 모델 훈련
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled, y_train)

# 정규화된 테스트 데이터에 대한 예측 수행
predictions = knn.predict(X_test_scaled)

# 예측 결과를 DataFrame으로 만들고, 파일로 저장
submission = pd.DataFrame({'ID': test_data.index, 'Class': predictions})
submission.to_csv('submission21.csv', index=False)
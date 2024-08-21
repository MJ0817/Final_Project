# 필요한 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. 데이터 불러오기
# 데이터 파일 경로 설정
file_path = '/Users/joon/Documents/GitHub/Final_Project/data/1.smoke_detection_iot.csv'
df = pd.read_csv(file_path)

# 데이터를 불러옵니다.
df = pd.read_csv(file_path)

# 2. 데이터 전처리
# 'Unnamed: 0' 열 제거 (인덱스 역할로 보이며 의미가 없으므로 제거)
df = df.drop(columns=['Unnamed: 0'])

# UTC 열에서 날짜 및 시간을 유의미한 특성으로 변환
df['UTC'] = pd.to_datetime(df['UTC'], unit='s')
df['Hour'] = df['UTC'].dt.hour
df['Day'] = df['UTC'].dt.day
df['Month'] = df['UTC'].dt.month
df['Year'] = df['UTC'].dt.year

# 원본 UTC 열 제거 (날짜와 시간에 대한 특성을 사용하므로 원본 타임스탬프는 제거)
df = df.drop(columns=['UTC'])

# 3. 데이터 탐색적 분석 (EDA)
# 데이터의 기본적인 통계량 확인
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# 상관계수 행렬 계산 및 시각화
correlation_matrix = df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 4. 모델링 및 평가
# 독립 변수와 종속 변수 정의
X = df.drop(columns=['Fire Alarm'])
y = df['Fire Alarm']

# 데이터셋을 학습용과 테스트용으로 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 테스트 세트에 대한 예측 수행
y_pred = model.predict(X_test)

# 모델 평가 결과 출력
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:\n", report)

# 5. 예측 결과 시각화
# 실제 값과 예측 값 비교 시각화
plt.figure(figsize=(10, 6))
sns.countplot(x=y_test, label="True Values", color='blue', alpha=0.6)
sns.countplot(x=y_pred, label="Predicted Values", color='red', alpha=0.3)
plt.legend()
plt.title("True vs Predicted Values")
plt.show()

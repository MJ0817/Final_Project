# 필요한 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. 데이터 불러오기 및 전처리
# 데이터 파일 경로 설정
file_path = '/Users/joon/Documents/GitHub/Final_Project/data/1.smoke_detection_iot.csv'
df = pd.read_csv(file_path)

# 불필요한 열 제거 및 UTC 시간 변환
df = df.drop(columns=['Unnamed: 0'])  # Unnamed: 0 열은 CSV 파일에서 자동으로 생성된 인덱스 열이므로 제거
df['UTC'] = pd.to_datetime(df['UTC'], unit='s')  # 유닉스 타임스탬프를 사람이 이해할 수 있는 날짜와 시간 형식으로 변환
df['Hour'] = df['UTC'].dt.hour  # 시간(hour) 정보 추출
df['Day'] = df['UTC'].dt.day  # 일(day) 정보 추출
df['Month'] = df['UTC'].dt.month  # 월(month) 정보 추출
df['Year'] = df['UTC'].dt.year  # 연(year) 정보 추출
df = df.drop(columns=['UTC'])  # 원본 UTC 열은 이제 필요 없으므로 제거

# 설명:
# 이 단계에서는 데이터를 불러와 분석에 필요 없는 열을 제거하고, 시간 데이터를 사람이 이해할 수 있는 형태로 변환했습니다.
# 시간을 연도, 월, 일, 시간으로 분리하여 나중에 시간 패턴 분석에 유용하도록 준비했습니다.

# 2. 탐색적 데이터 분석 (EDA)

# 상관관계 히트맵 시각화
correlation_matrix = df.corr()  # 모든 변수 간의 상관관계를 계산
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix of All Variables", fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.show()

# 설명:
# 이 히트맵은 데이터셋의 모든 변수 간 상관관계를 시각화한 것입니다. 각 셀의 색상은 두 변수 간의 상관관계 강도를 나타냅니다.
# 양의 상관관계는 빨간색에 가까운 색상으로, 음의 상관관계는 파란색에 가까운 색상으로 표시됩니다.
# 상관관계가 높은 변수는 서로 밀접하게 관련되어 있음을 의미하며, 이는 데이터 분석 및 모델링에 중요한 정보를 제공합니다.

# 3. 시간에 따른 데이터 시각화 (예시)

# TVOC (총 휘발성 유기 화합물) 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['TVOC[ppb]'], label='TVOC (ppb)', color='green')
plt.xlabel('Time (Hour)')
plt.ylabel('TVOC (ppb)')
plt.title('Total Volatile Organic Compounds (TVOC) Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# eCO2 (이산화탄소 농도) 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['eCO2[ppm]'], label='eCO2 (ppm)', color='red')
plt.xlabel('Time (Hour)')
plt.ylabel('eCO2 (ppm)')
plt.title('Equivalent CO2 (eCO2) Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Raw H2 (수소 농도) 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['Raw H2'], label='Raw H2', color='blue')
plt.xlabel('Time (Hour)')
plt.ylabel('Raw H2')
plt.title('Raw Hydrogen (H2) Levels Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Raw Ethanol (에탄올 농도) 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['Raw Ethanol'], label='Raw Ethanol', color='purple')
plt.xlabel('Time (Hour)')
plt.ylabel('Raw Ethanol')
plt.title('Raw Ethanol Levels Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Pressure (대기압) 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['Pressure[hPa]'], label='Pressure (hPa)', color='orange')
plt.xlabel('Time (Hour)')
plt.ylabel('Pressure (hPa)')
plt.title('Pressure Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# PM1.0과 PM2.5 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['PM1.0'], label='PM1.0 (μg/m³)', color='brown')
plt.plot(df['Hour'], df['PM2.5'], label='PM2.5 (μg/m³)', color='grey')
plt.xlabel('Time (Hour)')
plt.ylabel('Particulate Matter (PM)')
plt.title('Particulate Matter (PM1.0 and PM2.5) Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 설명:
# 각 그래프는 특정 환경 변수(TVOC, eCO2, Raw H2, Raw Ethanol, Pressure, PM1.0, PM2.5)의 시간에 따른 변화를 시각화합니다.
# 이를 통해 시간이 지남에 따라 이러한 변수들이 어떻게 변동하는지 쉽게 파악할 수 있습니다.
# 예를 들어, 특정 시간대에 TVOC나 eCO2 농도가 증가하는 패턴을 발견할 수 있습니다.

# 4. 모델링 및 평가

# 독립 변수와 종속 변수 설정
X = df.drop(columns=['Fire Alarm'])  # 종속 변수인 'Fire Alarm'을 제외한 나머지를 독립 변수로 설정
y = df['Fire Alarm']  # 'Fire Alarm'을 종속 변수로 설정

# 데이터셋을 학습용과 테스트용으로 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)  # 랜덤 포레스트 모델 생성
model.fit(X_train, y_train)  # 학습 데이터로 모델 학습

# 테스트 세트에 대한 예측 및 모델 평가
y_pred = model.predict(X_test)  # 테스트 데이터에 대한 예측 수행
accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
report = classification_report(y_test, y_pred)  # 분류 보고서 생성

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# 설명:
# 랜덤 포레스트 모델을 사용해 화재 경보(Fire Alarm)를 예측합니다.
# 데이터를 학습용과 테스트용으로 나누어, 모델의 예측 정확도와 성능을 평가합니다.
# 결과로 출력되는 정확도와 분류 보고서를 통해 모델의 성능을 확인할 수 있습니다.

# 5. 특성 중요도 시각화

importances = model.feature_importances_  # 각 변수의 중요도 계산
indices = importances.argsort()[::-1]  # 중요도를 기준으로 내림차순 정렬

plt.figure(figsize=(12, 8))
plt.title("Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90, fontsize=12)
plt.ylabel('Importance', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xlim([-1, X.shape[1]])
plt.show()

# 설명:
# 이 그래프는 모델이 예측을 할 때 어떤 변수를 가장 중요하게 사용했는지를 보여줍니다.
# 막대가 클수록 해당 변수가 예측에 미치는 영향이 크다는 의미입니다.
# 이를 통해 모델이 어떤 환경 변수를 중점적으로 고려하는지 파악할 수 있습니다.

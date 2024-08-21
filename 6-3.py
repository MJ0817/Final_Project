import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 데이터 로드
file_path = '/Users/joon/Documents/GitHub/Final_Project/data/1.smoke_detection_iot.csv'
df = pd.read_csv(file_path)

# 2. 데이터 전처리
df.isna().sum()
df = df[(df['Temperature[C]'] >= -50) & (df['Temperature[C]'] <= 100)]

# 3. 피처 엔지니어링
df['Temperature_MA'] = df['Temperature[C]'].rolling(window=60).mean()
df['Humidity_MA'] = df['Humidity[%]'].rolling(window=60).mean()
df['PM1.0_MA'] = df['PM1.0'].rolling(window=60).mean()
df['PM2.5_MA'] = df['PM2.5'].rolling(window=60).mean()

# 4. 데이터셋 분할
X = df[['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2',
        'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5',
        'NC0.5', 'NC2.5', 'Temperature_MA', 'Humidity_MA',
        'PM1.0_MA', 'PM2.5_MA']]
y = df['Fire Alarm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(cm)

# 7. 실시간 화재 예측
new_data = {
    'Temperature[C]': 27,
    'Humidity[%]': 45,
    'TVOC[ppb]': 500,
    'eCO2[ppm]': 450,
    'Raw H2': 160,
    'Raw Ethanol': 210,
    'Pressure[hPa]': 1010,
    'PM1.0': 12,
    'PM2.5': 25,
    'NC0.5': 550,
    'NC2.5': 260,
    'Temperature_MA': 26.5,
    'Humidity_MA': 44.0,
    'PM1.0_MA': 11.5,
    'PM2.5_MA': 24.5
}

new_df = pd.DataFrame([new_data])
prediction = model.predict(new_df)

print('Fire Detected' if prediction[0] == 1 else 'No Fire Detected')

# 8. 데이터 시각화 (깔끔하고 멋진 그래프)
# Seaborn 스타일 설정
sns.set(style="whitegrid")

# Temperature와 Fire Alarm의 관계 시각화
plt.figure(figsize=(14, 7))
plt.plot(df['Temperature[C]'], label='Temperature (C)', color='orange', linewidth=2)
plt.plot(df['Fire Alarm'] * 50, label='Fire Alarm', color='red', alpha=0.6, linewidth=2)
plt.title('Temperature and Fire Alarm over Time', fontsize=18)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Temperature (C)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Humidity와 Fire Alarm의 관계 시각화
plt.figure(figsize=(14, 7))
plt.plot(df['Humidity[%]'], label='Humidity (%)', color='blue', linewidth=2)
plt.plot(df['Fire Alarm'] * 100, label='Fire Alarm', color='red', alpha=0.6, linewidth=2)
plt.title('Humidity and Fire Alarm over Time', fontsize=18)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Humidity (%)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Confusion Matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', linewidths=2, linecolor='black',
            xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'], cbar=False)
plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.show()

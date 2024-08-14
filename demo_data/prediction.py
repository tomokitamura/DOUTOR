import os
import sys  # 追加
os.chdir("/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR")
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle  # スケーラーを保存・読み込みするためのモジュール
from demo_data import make_demodata

# 追加: コマンドライン引数からtargetを取得
if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = "ブレンドコーヒー"  # デフォルト値

save_data = True
comment = target

# 生成したデータを利用
X, y = make_demodata.generate_demo_data(start_date='2024-03-27',  # データの開始日
                       end_date='2050-08-11',  # データの終了日
                       winter_ice_ratio=(0.2, 0.3),  # 冬のアイスドリンク割合
                       summer_ice_ratio=(0.75, 0.8),  # 夏のアイスドリンク割合
                       weekday_sales_range=(200, 250),  # 平日のドリンク販売数範囲
                       weekend_sales_range=(250, 300),  # 休日のドリンク販売数範囲
                       rain_reduction_range=(0.8, 0.9),  # 雨の日の販売減少割合範囲
                       revenue_range=(100000, 200000))  # 売上高の範囲

if save_data == True:
    make_demodata.save_demo_data(X, y, '/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data', comment=comment)

# 従属変数と目的変数の設定
X_features = X.drop(columns=["date"])

y_target = y[target]

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# インデックスを保持
X_test_index = X_test.index

# データを標準化するためのスケーラーを初期化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 訓練データの標準化
X_train = scaler_X.fit_transform(X_train)

# テストデータの標準化（訓練データのスケーリングに基づく）
X_test = scaler_X.transform(X_test)

# 目的変数（y）の標準化
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# スケーラーの保存
scaler_X_path = "/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/scaler_X.pkl"
scaler_y_path = "/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/scaler_y.pkl"
with open(scaler_X_path, 'wb') as f:
    pickle.dump(scaler_X, f)
with open(scaler_y_path, 'wb') as f:
    pickle.dump(scaler_y, f)

# モデルの構築
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 目的変数を予測

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(input_shape=X_train.shape[1])

# EarlyStoppingのコールバックを作成
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# モデルの構築とトレーニング（EarlyStoppingを使用）
history = model.fit(X_train, y_train_scaled, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# モデルの保存
model_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/{target}_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

# テストデータでの予測
y_pred_scaled = model.predict(X_test)

# 予測結果を元のスケールに戻す
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# MAEの計算を元のスケールで行う
test_mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE (Original Scale): {test_mae}')

# カスタムスコアリング関数を定義
def custom_scorer(model, X, y):
    y_pred = model.predict(X)
    return -mean_absolute_error(y, y_pred)

# パーミュテーションインポータンスの計算
perm_importance = permutation_importance(model, X_test, y_test_scaled, n_repeats=10, random_state=42, scoring=custom_scorer)

# 重要度を持つ特徴量を取得
feature_names = X_features.columns
sorted_idx = perm_importance.importances_mean.argsort()

# プロット
plt.figure(figsize=(10, 8))
plt.barh(feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Permutation Importance of Features")
plt.show()

# 予測結果をデータフレームにまとめる
results_df = pd.DataFrame({
    'index': X_test_index,  # テストデータの元の日付に代わるインデックス
    f'{target}_actual': y_test,  # 実測値
    f'{target}_predicted': y_pred  # 予測値
})

# ファイル名の生成
a = datetime.datetime.now().strftime('%Y%m%d%H%M')
filename = f'demo_result_{comment}_{a}.csv'
full_path = os.path.join('/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data', filename)

# CSVとして保存
results_df.to_csv(full_path, index=False)
print(f'Results saved to {full_path}')
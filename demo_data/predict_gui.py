import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import make_demodata
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# GUIのセットアップ
window = tk.Tk()
window.title("売上予測システム")

# 月を選択するプルダウンメニュー
month_label = tk.Label(window, text="月:")
month_label.pack()
month_var = tk.StringVar(window)
month_var.set("1")  # デフォルト値
month_menu = tk.OptionMenu(window, month_var, *list(range(1, 13)))
month_menu.pack()

# 日を選択するプルダウンメニュー
day_label = tk.Label(window, text="日:")
day_label.pack()
day_var = tk.StringVar(window)
day_var.set("1")  # デフォルト値
day_menu = tk.OptionMenu(window, day_var, *list(range(1, 32)))
day_menu.pack()

# 最高気温を入力するエントリ
high_temp_label = tk.Label(window, text="最高気温:")
high_temp_label.pack()
high_temp_entry = tk.Entry(window)
high_temp_entry.pack()

# 最低気温を入力するエントリ
low_temp_label = tk.Label(window, text="最低気温:")
low_temp_label.pack()
low_temp_entry = tk.Entry(window)
low_temp_entry.pack()

# 天気を選択するプルダウンメニュー
weather_label = tk.Label(window, text="天気:")
weather_label.pack()
weather_var = tk.StringVar(window)
weather_var.set("晴")  # デフォルト値
weather_menu = tk.OptionMenu(window, weather_var, "晴", "曇り", "雨")
weather_menu.pack()

# 目的変数を選択するプルダウンメニュー
target_label = tk.Label(window, text="目的変数を選択してください:")
target_label.pack()
target_var = tk.StringVar(window)
target_var.set("ブレンドコーヒー")  # デフォルト値

# すべてのメニューを含む選択肢
menu_options = [
    "ブレンドコーヒー", "アイスコーヒー", "アメリカンコーヒー", "本日のおすすめコーヒー",
    "カフェ・ラテ", "豆乳ラテ", "ハニーカフェ・オレ", "沖縄黒糖ラテ",
    "カフェインレスコーヒー", "カフェインレスカフェ・ラテ", "ティー", "アイスティー",
    "ロイヤルミルクティー", "豆乳ティー", "国産みかんジュース", "青森県産りんごジュース",
    "カフェ・モカ", "ココア", "宇治抹茶ラテ", "キッチントルコライス",
    "ビーフシチュープレート", "焼きトマトのハンバーグドリア", "海老とペンネのグラタン",
    "十六穀米のミートドリア", "からあげ", "フライドポテト", "エビアボカドサンド",
    "ポンレスハム＆生ハムサンド", "キッチンチーズドッグ", "キッチンレタスドッグ",
    "キッチンジャーマンドッグ", "ピザトースト", "瀬戸内海産しらすとたらこ", "ナポリタン",
    "アイスドリンクの杯数", "ホットドリンクの杯数", "accurate_revenue", "revenue"
]

target_menu = tk.OptionMenu(window, target_var, *menu_options)
target_menu.pack()

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 目的変数を予測
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def generate_and_save_model(target):
    # 生成したデータを利用
    X, y = make_demodata.generate_demo_data(start_date='2024-03-27',  # データの開始日
                           end_date='2024-08-11',  # データの終了日
                           winter_ice_ratio=(0.2, 0.3),  # 冬のアイスドリンク割合
                           summer_ice_ratio=(0.75, 0.8),  # 夏のアイスドリンク割合
                           weekday_sales_range=(200, 250),  # 平日のドリンク販売数範囲
                           weekend_sales_range=(250, 300),  # 休日のドリンク販売数範囲
                           rain_reduction_range=(0.8, 0.9),  # 雨の日の販売減少割合範囲
                           revenue_range=(100000, 200000))  # 売上高の範囲

    X_features = X.drop(columns=["date"])
    y_target = y[target]

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # スケーラーの保存
    scaler_X_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/scaler_X_{target}.pkl"
    scaler_y_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/scaler_y_{target}.pkl"
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)

    # モデルの構築と保存
    model = build_model(input_shape=X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train_scaled, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    model_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/{target}_model.h5"
    model.save(model_path)
    print(f"Model for {target} saved to {model_path}")

def predict():
    try:
        target = target_var.get()
        model_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/{target}_model.h5"

        # モデルが存在しない場合は生成
        if not os.path.exists(model_path):
            generate_and_save_model(target)

        # モデルとスケーラーの読み込み
        model = tf.keras.models.load_model(model_path)
        scaler_X_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/scaler_X_{target}.pkl"
        scaler_y_path = f"/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/DOUTOR/demo_data/scaler_y_{target}.pkl"
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)

        # 入力値を取得
        month = int(month_var.get())
        day = int(day_var.get())
        high_temp = float(high_temp_entry.get())
        low_temp = float(low_temp_entry.get())
        weather = weather_var.get()

        # 天気のカテゴリカル変数の処理
        weather_dict = {"晴": 0, "曇り": 1, "雨": 2}
        weather = weather_dict[weather]

        # データの準備（weekdayは仮に0で初期化）
        X_new = np.array([[month, day, 0, high_temp, low_temp, weather]])

        # 標準化
        X_new_scaled = scaler_X.transform(X_new)

        # 予測
        y_pred_scaled = model.predict(X_new_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()[0]

        # 結果を整数値で表示
        result_label.config(text=f"{target}の予測: {int(y_pred)}", font=("Helvetica", 16, "bold"))
    except Exception as e:
        messagebox.showerror("エラー", f"予測中にエラーが発生しました: {e}")

# 予測ボタン
predict_button = tk.Button(window, text="予測", command=predict)
predict_button.pack()

# 結果表示ラベル
result_label = tk.Label(window, text="")
result_label.pack()

# メインループ
window.mainloop()
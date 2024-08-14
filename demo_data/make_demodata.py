import numpy as np
import pandas as pd
import datetime
import os

import numpy as np
import pandas as pd
import datetime
import os

def generate_demo_data(start_date='2022-01-01',  # データの開始日
                       end_date='2023-12-31',  # データの終了日
                       winter_ice_ratio=(0.3, 0.4),  # 冬のアイスドリンク割合
                       summer_ice_ratio=(0.8, 0.95),  # 夏のアイスドリンク割合
                       weekday_sales_range=(200, 230),  # 平日のドリンク販売数範囲
                       weekend_sales_range=(250, 280),  # 休日のドリンク販売数範囲
                       rain_reduction_range=(0.7, 0.9),  # 雨の日の販売減少割合範囲
                       revenue_range=(100000, 200000)):  # 売上高の範囲

    # 商品名と価格の辞書
    items = {
        "ブレンドコーヒー": 300,
        "アイスコーヒー": 300,
        "アメリカンコーヒー": 300,
        "本日のおすすめコーヒー": 380,
        "カフェ・ラテ": 390,
        "豆乳ラテ": 390,
        "ハニーカフェ・オレ": 410,
        "沖縄黒糖ラテ": 480,
        "カフェインレスコーヒー": 300,
        "カフェインレスカフェ・ラテ": 390,
        "ティー": 260,
        "アイスティー": 310,
        "ロイヤルミルクティー": 410,
        "豆乳ティー": 410,
        "国産みかんジュース": 440,
        "青森県産りんごジュース": 440,
        "カフェ・モカ": 440,
        "ココア": 400,
        "宇治抹茶ラテ": 480,
        "キッチントルコライス": 960,
        "ビーフシチュープレート": 960,
        "焼きトマトのハンバーグドリア": 940,
        "海老とペンネのグラタン": 890,
        "十六穀米のミートドリア": 870,
        "からあげ": 360,
        "フライドポテト": 290,
        "エビアボカドサンド": 550,
        "ポンレスハム＆生ハムサンド": 550,
        "キッチンチーズドッグ": 440,
        "キッチンレタスドッグ": 380,
        "キッチンジャーマンドッグ": 350,
        "ピザトースト": 450,
        "瀬戸内海産しらすとたらこ": 790,
        "ナポリタン": 790,
        # 必要に応じてケーキのデータも追加
    }

    # 日付範囲を生成
    dates = pd.date_range(start=start_date, end=end_date)
    n_samples = len(dates)
    
    # 月、日、曜日を抽出
    months = dates.month
    days = dates.day
    weekdays = dates.weekday
    
    # 天気の設定（晴:0, 曇り:1, 雨:2）
    weather = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
    
    # 気温の設定（月ごとの平均気温範囲を参考にランダムに生成）
    temperature_ranges = {
        1: (0, 10), 2: (2, 12), 3: (5, 15), 4: (10, 20),
        5: (15, 25), 6: (18, 28), 7: (22, 32), 8: (22, 32),
        9: (18, 28), 10: (12, 22), 11: (7, 17), 12: (2, 12)
    }
    high_temp = np.array([int(np.random.uniform(*temperature_ranges[month])) for month in months])
    low_temp = np.array([int(high - np.random.uniform(5, 10)) for high in high_temp])

    # ドリンク販売数の設定
    ice_drink_ratio = np.zeros(n_samples)
    hot_drink_ratio = np.zeros(n_samples)
    
    # グラデーション設定
    for i in range(n_samples):
        month = months[i]
        if month in [1, 2, 11, 12]:  # 冬
            ice_drink_ratio[i] = np.random.uniform(*winter_ice_ratio)
        elif month in [6, 7, 8]:  # 夏
            ice_drink_ratio[i] = np.random.uniform(*summer_ice_ratio)
        else:  # 春と秋: それぞれの範囲内でグラデーション
            ice_drink_ratio[i] = np.random.uniform(0.4, 0.6)
        hot_drink_ratio[i] = 1 - ice_drink_ratio[i]
    
    # 平日と休日でドリンクの販売数を設定
    total_drink_sales = np.where(weekdays < 5,  # 平日
                                 np.random.randint(*weekday_sales_range, size=n_samples),
                                 np.random.randint(*weekend_sales_range, size=n_samples))  # 休日

    ice_drink_sales = (total_drink_sales * ice_drink_ratio).astype(int)
    hot_drink_sales = (total_drink_sales * hot_drink_ratio).astype(int)
    
    # 天気が雨の日はドリンクの販売数を減少
    reduction_factor = np.where(weather == 2, np.random.uniform(*rain_reduction_range, size=n_samples), 1.0)
    ice_drink_sales = (ice_drink_sales * reduction_factor).astype(int)
    hot_drink_sales = (hot_drink_sales * reduction_factor).astype(int)
    
    # 商品販売数の設定（正規分布に基づく）
    menu_sales = {}
    for item, price in items.items():
        mean = price // 10  # 値段に基づいて平均を設定（仮のロジック）
        sales = np.clip(np.random.normal(mean, mean * 0.2, size=n_samples).astype(int), 1, mean * 2)
        # 天気の影響を反映
        sales = (sales * reduction_factor).astype(int)
        menu_sales[item] = sales
    
    # accurate_revenueの計算
    accurate_revenue = np.zeros(n_samples)
    for item, sales in menu_sales.items():
        accurate_revenue += sales * items[item]
    
    # revenueの計算 (指定された範囲の平均を中心とする正規分布)
    revenue_mean = np.mean(revenue_range)
    revenue_std = (revenue_range[1] - revenue_range[0]) / 6  # 3σの範囲に収まるように標準偏差を設定
    base_revenue = np.random.normal(revenue_mean, revenue_std, size=n_samples)
    base_revenue = np.clip(base_revenue, revenue_range[0], revenue_range[1])  # 指定範囲内にクリップ
    revenue = base_revenue * reduction_factor
    
    # データフレームの作成
    X = pd.DataFrame({
        'date': dates,
        'month': months,
        'day' : days,
        'weekday': weekdays,
        'high_temp': high_temp,
        'low_temp': low_temp,
        'weather': weather
    })
    
    y = pd.DataFrame({
        **menu_sales,
        'ice_drink_sales': ice_drink_sales,
        'hot_drink_sales': hot_drink_sales,
        'accurate_revenue': accurate_revenue.astype(int),
        'revenue': revenue.astype(int)
    })
    
    return X, y


def save_demo_data(X, y, path, comment):
    # 現在の時間から秒数を除いた形式で`a`を生成
    a = datetime.datetime.now().strftime('%Y%m%d%H%M')
    filename = f'demo_{comment}_{a}.csv'
    full_path = os.path.join(path, filename)

    # Xとyを結合して保存
    full_data = pd.concat([X, y], axis=1)
    full_data.to_csv(full_path, index=False)
    print(f'Data saved to {full_path}')

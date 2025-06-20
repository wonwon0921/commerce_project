import warnings
import pandas as pd
import sqlite3
import os

warnings.filterwarnings('ignore')  # 忽略警告訊息

# 連接 SQLite 資料庫
conn = sqlite3.connect(r"/Users/ben/Desktop/sqlite-練習題/amazon.db")  # 改成你的資料庫路徑

# 從資料庫讀取資料表 commerce_data，並指定 CustomerID 為字串型態
df = pd.read_sql_query("SELECT * FROM commerce_data", conn, dtype={'CustomerID': str})

# 關閉連線
conn.close()

# 查看前五筆資料
print(df.head())

# 查看資料維度
print("資料形狀:", df.shape)

# 查看欄位資訊與缺失值狀況
print(df.info())

# 計算每欄位缺失比例
missing_ratio = df.isnull().mean()
print("缺失比例:\n", missing_ratio)

# 移除 Description 欄位（如果存在）
if 'Description' in df.columns:
    df.drop(['Description'], axis=1, inplace=True)

# 再次查看前五筆確認刪除成功
print(df.head())

# 將 CustomerID 缺失值填補為 'Unknown'
df['CustomerID'] = df['CustomerID'].fillna('Unknown')

# 把空字串（空白）也轉成 'Unknown'
df['CustomerID'] = df['CustomerID'].replace(r'^\s*$', 'Unknown', regex=True)

# 計算重複行數
print("重複行數:", df.duplicated().sum())

# 去除重複行
df = df.drop_duplicates()

# 查看數值型欄位的描述統計
print(df.describe())

# 確認 Unknown 數量
print("CustomerID 中 'Unknown' 數量:", (df['CustomerID'] == 'Unknown').sum())

#RFM
# 1. 找出每個顧客最後一次消費日期
last_trans_date = df_buy.groupby('CustomerID')['Date'].max()

# 2. 設定參考日期（分析當天的下一天）
snapshot_date = df_buy['Date'].max() + pd.Timedelta(days=1)

# 3. 計算 Recency (距離參考日期的天數)
R = (snapshot_date - last_trans_date).dt.days

# 4. 計算 Frequency (消費頻率，訂單數)
F = df_buy.groupby('CustomerID')['InvoiceNo'].nunique()

# 5. 計算 Monetary (消費金額總和)
M = df_buy.groupby('CustomerID')['Total'].sum()

# 6. 合併成 RFM 表
rfm = pd.DataFrame({
    'Recency': R,
    'Frequency': F,
    'Monetary': M
})

# 7. 使用分位數過濾極端值（例如排除上下1%的極端值）
lower_q = 0.05
upper_q = 0.95

rfm_filtered = rfm[
    (rfm['Frequency'] >= rfm['Frequency'].quantile(lower_q)) &
    (rfm['Frequency'] <= rfm['Frequency'].quantile(upper_q)) &
    (rfm['Monetary'] >= rfm['Monetary'].quantile(lower_q)) &
    (rfm['Monetary'] <= rfm['Monetary'].quantile(upper_q))
].copy()

print(f"原始筆數: {len(rfm)}, 過濾後筆數: {len(rfm_filtered)}")

# 8. 查看過濾後的描述統計
print("過濾後 RFM 描述統計:")
print(rfm_filtered.describe())

# 9. 確認過濾後前幾筆
print("\n過濾後 RFM 資料預覽:")
print(rfm_filtered.head())

# Recency反向百分比排名，越近越大分數（5分為最高）
rfm_filtered['R_Score'] = (1 - rfm_filtered['Recency'].rank(pct=True)) * 5
rfm_filtered['R_Score'] = rfm_filtered['R_Score'].round().astype(int).clip(1,5)

# Frequency正向百分比排名，越頻繁分數越高
rfm_filtered['F_Score'] = rfm_filtered['Frequency'].rank(pct=True) * 5
rfm_filtered['F_Score'] = rfm_filtered['F_Score'].round().astype(int).clip(1,5)

# Monetary正向百分比排名，消費金額越大分數越高
rfm_filtered['M_Score'] = rfm_filtered['Monetary'].rank(pct=True) * 5
rfm_filtered['M_Score'] = rfm_filtered['M_Score'].round().astype(int).clip(1,5)

# 合併成 RFM 分數字串
rfm_filtered['RFM_Score'] = rfm_filtered['R_Score'].astype(str) + \
                            rfm_filtered['F_Score'].astype(str) + \
                            rfm_filtered['M_Score'].astype(str)

print(rfm_filtered[['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']].head(10))
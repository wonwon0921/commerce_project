import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

# 連接 SQLite 資料庫
conn = sqlite3.connect(r"/Users/ben/Desktop/python/practice/data_practice/amazon.db")

# 從資料庫讀取資料表 commerce_data，並指定 CustomerID 為字串型態
df = pd.read_sql_query("SELECT * FROM commerce_data", conn, dtype={'CustomerID': str})

# 關閉連線
conn.close()
# 排除退貨單據
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
# 查看前五筆資料
print(df.head())

# 查看資料維度
print("資料形狀:", df.shape)

# 查看欄位資訊與缺失值狀況
print(df.info())

# 計算每欄位缺失比例
missing_ratio = df.isnull().mean()
print("缺失比例:\n", missing_ratio)

# 移除 Description 欄位
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

# 數值欄位轉換
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# 過濾異常資料
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
# 拆解時間欄位並轉成 datetime
df[['Date', 'Time']] = df['InvoiceDate'].str.split(' ', expand=True)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
df.drop(columns=['InvoiceDate'], inplace=True)

# 計算每筆交易總額
df['Total'] = df['Quantity'] * df['UnitPrice']

# 確認結果
print("清理後資料筆數:", len(df))
print("CustomerID 'Unknown' 數量:", (df['CustomerID'] == 'Unknown').sum())
print(df[['Quantity', 'UnitPrice', 'Total']].describe())

# 新增欄位判斷是否會員
df['MemberType'] = df['CustomerID'].apply(lambda x: 'Non-Member' if x == 'Unknown' else 'Member')

# 計算各會員類型的訂單數（InvoiceNo 去重計數）
order_counts = df.groupby(['MemberType', 'InvoiceNo']).size().reset_index(name='OrderLines')
order_counts_per_member = order_counts.groupby('MemberType').size()

# 計算各會員類型總購買金額
total_spent = df.groupby('MemberType')['Total'].sum()

# 合併成表
summary = pd.DataFrame({
    'OrderCount': order_counts_per_member,
    'TotalSpent': total_spent
})

print("會員 vs 非會員 購買行為統計：")
print(summary)

# 日期轉換
df['Date'] = pd.to_datetime(df['Date'])

# 訂單月份
df['OrderMonth'] = df['Date'].dt.to_period('M')

# 首次購買月份 CohortMonth
cohort = df.groupby('CustomerID')['OrderMonth'].min().reset_index()
cohort.columns = ['CustomerID', 'CohortMonth']

# 避免重複欄位衝突
if 'CohortMonth' in df.columns:
    df = df.drop(columns=['CohortMonth'])

df = df.merge(cohort, on='CustomerID')

# 建立 CohortIndex
df['CohortIndex'] = (df['OrderMonth'].dt.year - df['CohortMonth'].dt.year) * 12 + \
                    (df['OrderMonth'].dt.month - df['CohortMonth'].dt.month)

# 每 cohort 每月的活躍顧客數
cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()

# 每 cohort 初始人數
cohort_size = cohort_data[cohort_data['CohortIndex'] == 0][['CohortMonth', 'CustomerID']]
cohort_size.columns = ['CohortMonth', 'CohortSize']
cohort_data = cohort_data.merge(cohort_size, on='CohortMonth')

# 計算留存率
cohort_data['RetentionRate'] = cohort_data['CustomerID'] / cohort_data['CohortSize']
cohort_data['CohortMonth'] = cohort_data['CohortMonth'].astype(str)

# 過濾觀察期不足的 cohort
min_observed_months = 3
cohort_validity = df.groupby('CohortMonth')['CohortIndex'].max().reset_index()
cohort_validity.columns = ['CohortMonth', 'MaxIndex']
valid_cohorts = cohort_validity[cohort_validity['MaxIndex'] >= min_observed_months]['CohortMonth'].astype(str)
cohort_data = cohort_data[cohort_data['CohortMonth'].isin(valid_cohorts)]

# 建立留存率 & 留存人數表格
retention_rate_table = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='RetentionRate')
retention_count_table = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

# 平均回購率（剔除 0 月）
kpi_retention = retention_rate_table.iloc[:, 1:].mean().mean()
print(f"\n回購率 KPI（平均留存率，不含第 0 月）: {kpi_retention:.2%}")

# 畫留存率熱力圖
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 8))
sns.heatmap(retention_rate_table, annot=True, fmt='.0%', cmap='Blues')
plt.title('Cohort 留存率（%）')
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.tight_layout()
plt.show()

# 畫留存人數熱力圖
plt.figure(figsize=(12, 8))
sns.heatmap(retention_count_table, annot=True, fmt='.0f', cmap='Purples')
plt.title('Cohort 留存人數')
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.tight_layout()
plt.show()
# RFM
# 1. 找出每個顧客最後一次消費日期
last_trans_date = df.groupby('CustomerID')['Date'].max()

# 2. 設定參考日期（分析當天的下一天）
snapshot_date = df['Date'].max() + pd.Timedelta(days=1)

# 3. 計算 Recency (距離參考日期的天數)
R = (snapshot_date - last_trans_date).dt.days

# 4. 計算 Frequency (消費頻率，訂單數)
F = df.groupby('CustomerID')['InvoiceNo'].nunique()

# 5. 計算 Monetary (消費金額總和)
M = df.groupby('CustomerID')['Total'].sum()

# 6. 合併成 RFM 表
rfm = pd.DataFrame({
    'Recency': R,
    'Frequency': F,
    'Monetary': M
})

# 7. 使用分位數過濾極端值
lower_q = 0.05
upper_q = 0.95

rfm_filtered = rfm[
    (rfm['Frequency'] >= rfm['Frequency'].quantile(lower_q)) &
    (rfm['Frequency'] <= rfm['Frequency'].quantile(upper_q)) &
    (rfm['Recency'] >= rfm['Recency'].quantile(lower_q)) &
    (rfm['Recency'] <= rfm['Recency'].quantile(upper_q)) &
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

rfm_filtered.to_csv("rfm_filtered.csv")

print(rfm_filtered[['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']].head(10))
# 畫圖 + 匯出 HTML
fig = px.histogram(rfm_filtered, x='RFM_Score', color='RFM_Score',
                   title='RFM Score 分布',
                   labels={'RFM_Score': 'RFM 分數'},
                   color_discrete_sequence=px.colors.sequential.Viridis)

fig.update_layout(xaxis=dict(categoryorder='category ascending'))
html_path = "/Users/ben/Desktop/rfm_score_plot.html"
fig.write_html(html_path)
webbrowser.open(html_path)

# 讀取 RFM 檔案
rfm = pd.read_csv("/Users/ben/Downloads/rfm_filtered.csv")

# 取出 RFM 數值欄位
rfm_values = rfm[['Recency', 'Frequency', 'Monetary']].copy()

# 標準化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_values)

# 使用標準化過的 rfm_scaled 進行 Elbow Method 分析
sse = []  # 存每個 k 對應的 inertia（SSE）

# 測試 k 值從 1 到 16
K_range = range(1, 16)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

# 設定字體為 SimHei
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 繪製 Elbow 曲線
plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, marker='o')
plt.xticks(K_range)
plt.xlabel('群數 k')
plt.ylabel('總體內平方誤差 (SSE)')
plt.title('Elbow Method 找最佳群數')
plt.grid(True)
plt.tight_layout()
plt.show()
# 執行 KMeans 分群
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 計算每群的平均值與數量
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
cluster_summary['Count'] = rfm['Cluster'].value_counts().sort_index()
print("每群統計摘要：\n", cluster_summary)

# 匯出 CSV（含分群結果）
rfm.to_csv("/Users/ben/Desktop/rfm_kmeans_segmented.csv", index=False)

# 畫 3D 分群圖
fig = px.scatter_3d(rfm,
                    x='Recency', y='Frequency', z='Monetary',
                    color=rfm['Cluster'].astype(str),
                    title='KMeans RFM 分群（4群）',
                    labels={'Cluster': '群組'},
                    hover_data=['CustomerID', 'RFM_Score'])

# 匯出圖為 HTML 並自動打開
fig.write_html("/Users/ben/Desktop/rfm_kmeans_3d.html")
webbrowser.open("/Users/ben/Desktop/rfm_kmeans_3d.html")

print(rfm.columns)
print(rfm.head())

desc_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].agg(
    ['count', 'mean', 'median', 'std', 'min', 'max']
).round(2)

print(desc_stats)

for col in ['Recency', 'Frequency', 'Monetary']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Cluster', y=col, data=rfm)
    plt.title(f'Cluster vs {col}')
    plt.show()
# 將 CustomerID 統一轉成字串型別（
df['CustomerID'] = df['CustomerID'].astype(str)
rfm['CustomerID'] = rfm['CustomerID'].astype(str)

# 把分群結果合併回原始 df（使用 CustomerID 做關聯）
df = df.merge(rfm[['CustomerID', 'Cluster']], on='CustomerID', how='inner')

# 建立月份欄位
df['Month'] = df['Date'].dt.to_period('M')
# 計算每個群集、每月的總消費金額
monthly_cluster = df.groupby(['Cluster', 'Month'])['Total'].sum().reset_index()

# 繪圖
plt.figure(figsize=(12, 6))
for cluster in sorted(monthly_cluster['Cluster'].unique()):
    cluster_data = monthly_cluster[monthly_cluster['Cluster'] == cluster]
    plt.plot(cluster_data['Month'].astype(str), cluster_data['Total'], label=f'Cluster {cluster}')

plt.title("各群客戶每月總消費金額變化")
plt.xlabel("月份")
plt.ylabel("總消費金額")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 確保 df 已含 Cluster 欄位
print(df['Cluster'].value_counts())

# 建立一個空的 dict 儲存每個群組的 Top5 商品資料
top5_items_per_cluster = {}

# 儲存各群組的分析結果（包含總銷售、人均銷售、Top5熱銷商品佔比）
top5_ratio_result = []

# 計算每群的獨立客戶數（人數）
customer_count_per_cluster = df.groupby('Cluster')['CustomerID'].nunique()

for cluster in sorted(df['Cluster'].unique()):
    cluster_df = df[df['Cluster'] == cluster]

    # 總銷售金額
    total_sales = cluster_df['Total'].sum()

    # 每商品銷售彙總
    item_sales = (
        cluster_df
        .groupby('StockCode')['Total']
        .sum()
        .reset_index()
        .sort_values(by='Total', ascending=False)
    )

    # 取前5熱銷商品
    top5_items = item_sales.head(5)
    top5_items_per_cluster[cluster] = top5_items

    # Top5 商品銷售金額
    top5_total_sales = top5_items['Total'].sum()

    # Top5 熱銷商品佔比
    top5_ratio = top5_total_sales / total_sales * 100 if total_sales > 0 else 0

    # 人均銷售金額
    avg_sales_per_customer = total_sales / customer_count_per_cluster.loc[cluster]

    top5_ratio_result.append({
        'Cluster': cluster,
        'CustomerCount': customer_count_per_cluster.loc[cluster],
        'TotalSales': total_sales,
        'AvgSalesPerCustomer': round(avg_sales_per_customer, 2),
        'Top5Sales': top5_total_sales,
        'Top5Ratio(%)': round(top5_ratio, 2)
    })

# 轉成 DataFrame 顯示
top5_ratio_df = pd.DataFrame(top5_ratio_result)

print("各群組分析彙整：")
print(top5_ratio_df)

# 印出各群熱銷商品 TOP 5
for cluster, top5_df in top5_items_per_cluster.items():
    print(f"\nCluster {cluster} 的熱銷商品 TOP 5：")
    print(top5_df)

# 繪製人均銷售金額長條圖
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='AvgSalesPerCustomer', data=top5_ratio_df, palette='viridis')
plt.title('各群人均銷售金額')
plt.ylabel('人均銷售金額')
plt.xlabel('群組')
plt.tight_layout()
plt.show()

# 繪製 Top5 熱銷商品銷售佔比長條圖
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='Top5Ratio(%)', data=top5_ratio_df, palette='magma')
plt.title('各群 Top5 熱銷商品銷售佔比 (%)')
plt.ylabel('Top5 銷售佔比 (%)')
plt.xlabel('群組')
plt.tight_layout()
plt.show()
# 計算每位客戶購買次數
customer_purchases = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
customer_purchases.columns = ['CustomerID', 'PurchaseCount']

# 定義回購客戶(購買次數 > 2)
customer_purchases['IsRepeat'] = (customer_purchases['PurchaseCount'] > 3).astype(int)

# 刪除已有的 IsRepeat 欄位，避免合併衝突
if 'IsRepeat' in rfm.columns:
    rfm = rfm.drop(columns=['IsRepeat'])

# 合併回購標記
rfm = rfm.merge(customer_purchases[['CustomerID', 'IsRepeat']], on='CustomerID', how='left')

# 計算群組回購率
repeat_rate_by_cluster = rfm.groupby('Cluster')['IsRepeat'].mean().round(4) * 100
print("\n各群組回購率：\n", repeat_rate_by_cluster)
# CLV
rfm['CLV'] = rfm['Monetary'] * 0.5
clv_by_cluster = rfm.groupby('Cluster')['CLV'].mean().round(4)
print("CLV（毛利 50%）:\n", clv_by_cluster)
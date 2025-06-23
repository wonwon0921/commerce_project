import os
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

# 建立 outputs 目錄
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM commerce_data", conn, dtype={'CustomerID': str})
    conn.close()
    return df

def preprocess_data(df):
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()  # 排除退貨單
    df = df.drop_duplicates()
    # 僅保留有 CustomerID 的資料（即會員）
    df = df[df['CustomerID'].notna()]
    df = df[df['CustomerID'].str.strip() != '']
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df[['Date', 'Time']] = df['InvoiceDate'].str.split(' ', expand=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df['Total'] = df['Quantity'] * df['UnitPrice']
    df['MemberType'] = df['CustomerID'].apply(lambda x: 'Non-Member' if x == 'Unknown' else 'Member')
    df = df.drop(columns=['InvoiceDate', 'Time'], errors='ignore')

    return df

def member_vs_nonmember_analysis(df):
    summary = df.groupby('MemberType').agg(
        OrderCount=('InvoiceNo', 'nunique'),
        TotalSpent=('Total', 'sum')
    )
    print(summary)

def cohort_analysis(df):
    df['OrderMonth'] = df['Date'].dt.to_period('M')
    cohort = df.groupby('CustomerID')['OrderMonth'].min().reset_index()
    cohort.columns = ['CustomerID', 'CohortMonth']
    df = df.merge(cohort, on='CustomerID')
    df['CohortIndex'] = (df['OrderMonth'].dt.year - df['CohortMonth'].dt.year) * 12 + \
                        (df['OrderMonth'].dt.month - df['CohortMonth'].dt.month)
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_size = cohort_data[cohort_data['CohortIndex'] == 0][['CohortMonth', 'CustomerID']]
    cohort_size.columns = ['CohortMonth', 'CohortSize']
    cohort_data = cohort_data.merge(cohort_size, on='CohortMonth')
    cohort_data['RetentionRate'] = cohort_data['CustomerID'] / cohort_data['CohortSize']
    cohort_data['CohortMonth'] = cohort_data['CohortMonth'].astype(str)
    min_observed_months = 3
    cohort_validity = df.groupby('CohortMonth')['CohortIndex'].max().reset_index()
    cohort_validity.columns = ['CohortMonth', 'MaxIndex']
    valid_cohorts = cohort_validity[cohort_validity['MaxIndex'] >= min_observed_months]['CohortMonth'].astype(str)
    cohort_data = cohort_data[cohort_data['CohortMonth'].isin(valid_cohorts)]

    retention_rate_table = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='RetentionRate')
    retention_count_table = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

    kpi_retention = retention_rate_table.iloc[:, 1:].mean().mean()
    print(f"\n回購率 KPI（平均留存率，不含第 0 月）: {kpi_retention:.2%}")

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8))
    sns.heatmap(retention_rate_table, annot=True, fmt='.0%', cmap='Blues')
    plt.title('Cohort 留存率（%）')
    plt.ylabel('Cohort Month')
    plt.xlabel('Months Since First Purchase')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cohort_retention_rate_heatmap.png")
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(retention_count_table, annot=True, fmt='.0f', cmap='Purples')
    plt.title('Cohort 留存人數')
    plt.ylabel('Cohort Month')
    plt.xlabel('Months Since First Purchase')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cohort_retention_count_heatmap.png")
    plt.show()

def score_rfm(rfm):
    rfm['R_Score'] = (1 - rfm['Recency'].rank(pct=True)) * 5
    rfm['F_Score'] = rfm['Frequency'].rank(pct=True) * 5
    rfm['M_Score'] = rfm['Monetary'].rank(pct=True) * 5
    for col in ['R_Score', 'F_Score', 'M_Score']:
        rfm[col] = rfm[col].round().astype(int).clip(1, 5)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    return rfm
def print_rfm_quantiles(rfm_filtered):
    print("\nRFM 各分位數（每 20%，對應 5 分制）")

    quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for col in ['Recency', 'Frequency', 'Monetary']:
        print(f"\n{col} 分位數：")
        print(rfm_filtered[col].quantile(quantiles).round(2))

def rfm_analysis(df):
    lower_q = 0.05
    upper_q = 0.95
    snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
    R = (snapshot_date - df.groupby('CustomerID')['Date'].max()).dt.days
    F = df.groupby('CustomerID')['InvoiceNo'].nunique()
    M = df.groupby('CustomerID')['Total'].sum()
    rfm = pd.DataFrame({'Recency': R, 'Frequency': F, 'Monetary': M})

    # 只針對 Recency 進行上界過濾，避免過濾掉活躍客戶
    rfm_filtered = rfm[
        (rfm['Recency'] <= rfm['Recency'].quantile(upper_q)) &
        (rfm['Frequency'] >= rfm['Frequency'].quantile(lower_q)) &
        (rfm['Frequency'] <= rfm['Frequency'].quantile(upper_q)) &
        (rfm['Monetary'] >= rfm['Monetary'].quantile(lower_q)) &
        (rfm['Monetary'] <= rfm['Monetary'].quantile(upper_q))
    ].copy()

    # 新增對數轉換欄位（避免 log(0) 用 log1p）
    rfm_filtered['Recency_log'] = np.log1p(rfm_filtered['Recency'])
    rfm_filtered['Monetary_log'] = np.log1p(rfm_filtered['Monetary'])
    print_rfm_quantiles(rfm_filtered)
    print(f"原始顧客數: {len(rfm)}, 過濾後顧客數: {len(rfm_filtered)}")
    print("\n過濾後 RFM 描述統計：")
    print(rfm_filtered.describe())

    rfm_filtered = score_rfm(rfm_filtered)
    rfm_filtered.reset_index(inplace=True)  # 將 CustomerID 從 index 變欄位

    print("\nRFM 分數預覽：")
    print(rfm_filtered[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']].head(10))
    fig = px.histogram(
        rfm_filtered,
        x='RFM_Score',
        color='RFM_Score',
        title='RFM Score 分布',
        labels={'RFM_Score': 'RFM 分數'},
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(xaxis=dict(categoryorder='category ascending'))

    rfm_html = f"{OUTPUT_DIR}/rfm_score_distribution.html"
    fig.write_html(rfm_html)
    webbrowser.open(rfm_html)

    rfm_filtered.to_csv(f"{OUTPUT_DIR}/rfm_filtered.csv", index=False)

    return rfm_filtered
def kmeans_clustering(rfm):
    # 使用對數轉換的欄位來分群
    rfm_values = rfm[['Recency_log', 'Frequency', 'Monetary_log']].copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_values)

    sse = []
    K_range = range(1, 16)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        sse.append(kmeans.inertia_)

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, sse, marker='o')
    plt.xticks(K_range)
    plt.xlabel('群數 k')
    plt.ylabel('總體內平方誤差 (SSE)')
    plt.title('Elbow Method 找最佳群數')
    plt.grid(True)
    plt.tight_layout()
    elbow_path = f"{OUTPUT_DIR}/elbow_method.png"
    plt.savefig(elbow_path)
    plt.show()

    # 使用 4 群
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    cluster_summary['Count'] = rfm['Cluster'].value_counts().sort_index()
    print("每群統計摘要：\n", cluster_summary)

    rfm_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].agg(['mean', 'std', 'min', 'max']).round(2)
    print("\n各群組 RFM 描述性統計（平均、標準差、最小值、最大值）：")
    print(rfm_stats)

    rfm.to_csv(f"{OUTPUT_DIR}/rfm_kmeans_segmented.csv", index=False)

    fig = px.scatter_3d(rfm,
                        x='Recency', y='Frequency', z='Monetary',
                        color=rfm['Cluster'].astype(str),
                        title='KMeans RFM 分群（4群）',
                        labels={'Cluster': '群組'},
                        hover_data=['CustomerID', 'RFM_Score'])
    rfm_3d_html = f"{OUTPUT_DIR}/rfm_kmeans_3d.html"
    fig.write_html(rfm_3d_html)
    webbrowser.open(rfm_3d_html)

    return rfm

def rfm_cluster_visualizations(rfm):
    #  銷售金額佔比 (圓餅圖)
    cluster_sales = rfm.groupby('Cluster')['Monetary'].sum().reset_index()
    cluster_sales['SalesShare'] = cluster_sales['Monetary'] / cluster_sales['Monetary'].sum()

    plt.figure(figsize=(7, 7))
    plt.pie(cluster_sales['SalesShare'], labels=cluster_sales['Cluster'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
    plt.title('各群銷售金額佔比')
    plt.axis('equal')
    plt.savefig(f"{OUTPUT_DIR}/sales_share_pie.png")
    plt.show()

    #  客戶比例 (長條圖)
    cluster_counts = rfm['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'CustomerCount']
    cluster_counts['Ratio'] = cluster_counts['CustomerCount'] / cluster_counts['CustomerCount'].sum() * 100

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Cluster', y='Ratio', data=cluster_counts, palette='Set2')
    plt.title('各群客戶數比例 (%)')
    plt.ylabel('比例 (%)')
    plt.xlabel('群組')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/customer_ratio_bar.png")
    plt.show()

def cluster_sales_analysis(df, rfm):
    df['CustomerID'] = df['CustomerID'].astype(str)
    rfm['CustomerID'] = rfm['CustomerID'].astype(str)

    print("原始 df 客戶數:", df['CustomerID'].nunique())
    print("rfm 資料客戶數:", rfm['CustomerID'].nunique())

    # 僅保留在 RFM 分群中的客戶
    df_rfm = df[df['CustomerID'].isin(rfm['CustomerID'])].copy()

    # 合併分群結果
    df_rfm = df_rfm.merge(rfm[['CustomerID', 'Cluster']], on='CustomerID', how='inner')

    print("合併後客戶數:", df_rfm['CustomerID'].nunique())

    # 1. 準備時間序列數據
    df_rfm['Month'] = df_rfm['Date'].dt.to_period('M').astype(str)
    monthly_cluster = df_rfm.groupby(['Cluster', 'Month'])['Total'].sum().reset_index()

    # 2. 篩選指定時間範圍 (2010-12 至 2011-12)
    time_range = monthly_cluster[
        (monthly_cluster['Month'] >= '2010-12') &
        (monthly_cluster['Month'] <= '2011-12')
        ].copy()

    # 3. 時間序列描述性統計 (按月分組)
    print("各群組按月消費趨勢統計：")
    trend_stats = time_range.groupby(['Cluster', 'Month'])['Total'].agg('sum').round(2)
    print(trend_stats)

    # 4. 繪製趨勢折線圖
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=time_range,
        x='Month',
        y='Total',
        hue='Cluster',
        style='Cluster',
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette='viridis'
    )

    plt.title('各群組每月消費趨勢 (2010/12 - 2011/12)', fontsize=14, pad=20)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('總消費金額', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='群組', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 標記特殊月份
    for cluster in time_range['Cluster'].unique():
        cluster_data = time_range[time_range['Cluster'] == cluster]
        max_month = cluster_data.loc[cluster_data['Total'].idxmax(), 'Month']
        max_value = cluster_data['Total'].max()
        plt.annotate(f'Max: {max_value:,.0f}',
                     xy=(max_month, max_value),
                     xytext=(10, 10),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cluster_monthly_trends.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 進階統計分析
    print("\n進階趨勢分析：")
    for cluster in sorted(time_range['Cluster'].unique()):
        cluster_data = time_range[time_range['Cluster'] == cluster].copy()

        # 計算月環比成長率
        cluster_data['Growth'] = cluster_data['Total'].pct_change() * 100

        # 排除2011-12月計算CV
        cluster_data_cv = cluster_data[cluster_data['Month'] != '2011-12']
        cv = cluster_data_cv['Total'].std() / cluster_data_cv['Total'].mean()

        # 輸出關鍵指標
        print(f"\n群組 {cluster} 趨勢特徵：")
        print(f"平均月消費（全部月份）: {cluster_data['Total'].mean():,.0f}")
        print(
            f"最高成長月份: {cluster_data.loc[cluster_data['Growth'].idxmax(), 'Month']} ({cluster_data['Growth'].max():.1f}%)")
        print(f"消費波動率 (CV，排除2011-12): {cv:.2f}")
    plt.figure(figsize=(12, 6))
    for cluster in sorted(monthly_cluster['Cluster'].unique()):
        cluster_data = monthly_cluster[monthly_cluster['Cluster'] == cluster]
        plt.plot(cluster_data['Month'].astype(str), cluster_data['Total'], label=f'Cluster {cluster}')

    # 各群總銷售額與佔比
    cluster_sales_share = df_rfm.groupby('Cluster')['Total'].sum().reset_index()
    total_sales_all = cluster_sales_share['Total'].sum()
    cluster_sales_share['SalesShare(%)'] = cluster_sales_share['Total'] / total_sales_all * 100


    # 排序並顯示結果
    cluster_sales_share = cluster_sales_share.sort_values(by='SalesShare(%)', ascending=False)
    print("\n各群佔總銷售額比例：")
    print(cluster_sales_share[['Cluster', 'Total', 'SalesShare(%)']])

    # 畫圖視覺化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='SalesShare(%)', data=cluster_sales_share, palette='Set2')
    plt.title('各群客戶佔總銷售額比例 (%)')
    plt.ylabel('銷售佔比 (%)')
    plt.xlabel('客戶群組')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    print("\n各群交易筆數分布：")
    print(df_rfm['Cluster'].value_counts())

    print("\n各群獨立客戶數分布：")
    print(df_rfm.groupby('Cluster')['CustomerID'].nunique())

    # 計算各群客戶數（獨立客戶數）
    customer_count_per_cluster = df_rfm.groupby('Cluster')['CustomerID'].nunique()

    top5_items_per_cluster = []
    top5_ratio_result = []

    for cluster in sorted(df_rfm['Cluster'].unique()):
        cluster_df = df_rfm[df_rfm['Cluster'] == cluster]

        total_sales = cluster_df['Total'].sum()

        # 每群商品銷售總額
        item_sales = cluster_df.groupby('StockCode')['Total'].sum().reset_index().sort_values(by='Total', ascending=False)

        # Top5 商品
        top5_items = item_sales.head(5)
        top5_items_per_cluster.append((cluster, top5_items))

        top5_total_sales = top5_items['Total'].sum()

        # Top5 銷售佔比
        top5_ratio = top5_total_sales / total_sales * 100 if total_sales > 0 else 0

        # 人均銷售額
        avg_sales_per_customer = total_sales / customer_count_per_cluster.loc[cluster]

        top5_ratio_result.append({
            'Cluster': cluster,
            'CustomerCount': customer_count_per_cluster.loc[cluster],
            'TotalSales': total_sales,
            'AvgSalesPerCustomer': round(avg_sales_per_customer, 2),
            'Top5Sales': top5_total_sales,
            'Top5Ratio(%)': round(top5_ratio, 2)
        })

    top5_ratio_df = pd.DataFrame(top5_ratio_result)
    print("\n各群組銷售分析彙整：")
    print(top5_ratio_df)

    for cluster, top5_df in top5_items_per_cluster:
        print(f"\nCluster {cluster} 熱銷商品 TOP 5：")
        print(top5_df)

    # 繪製人均銷售額與Top5銷售佔比
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='AvgSalesPerCustomer', data=top5_ratio_df, palette='viridis')
    plt.title('各群人均銷售金額')
    plt.ylabel('人均銷售金額')
    plt.xlabel('群組')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Top5Ratio(%)', data=top5_ratio_df, palette='magma')
    plt.title('各群 Top5 熱銷商品銷售佔比 (%)')
    plt.ylabel('Top5 銷售佔比 (%)')
    plt.xlabel('群組')
    plt.tight_layout()
    plt.show()

    # 計算回購率（購買超過3次算回購）
    customer_purchases = df_rfm.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    customer_purchases.columns = ['CustomerID', 'PurchaseCount']
    customer_purchases['IsRepeat'] = (customer_purchases['PurchaseCount'] > 3).astype(int)

    if 'IsRepeat' in rfm.columns:
        rfm = rfm.drop(columns=['IsRepeat'])

    rfm = rfm.merge(customer_purchases[['CustomerID', 'IsRepeat']], on='CustomerID', how='left')

    repeat_rate_by_cluster = rfm.groupby('Cluster')['IsRepeat'].mean().round(4) * 100
    print("\n回購率 (購買超過3次占比) by Cluster:")
    print(repeat_rate_by_cluster)


def descriptive_stats(df):
    print("\n資料形狀:", df.shape)
    print("欄位資訊:")
    print(df.info())
    print("\n缺失比例:")
    print(df.isna().mean())
    print("\nCustomerID 中 'Unknown' 數量:", (df['CustomerID'] == 'Unknown').sum())
    print("\n數值描述統計:")
    print(df[['Quantity', 'UnitPrice', 'Total']].describe())

def main():
    DB_PATH = "/Users/ben/Desktop/python/practice/data_practice/amazon.db"

    df = load_data(DB_PATH)
    df = preprocess_data(df)

    descriptive_stats(df)

    member_vs_nonmember_analysis(df)

    cohort_analysis(df)

    rfm = rfm_analysis(df)

    rfm = kmeans_clustering(rfm)

    rfm_cluster_visualizations(rfm)

    cluster_sales_analysis(df, rfm)



if __name__ == "__main__":
    main()
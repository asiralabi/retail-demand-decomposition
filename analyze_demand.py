import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

# Load Data
def load_data():
    calendar = pd.read_csv('Datasets/calendar.csv')
    sales = pd.read_csv('Datasets/sales_train_evaluation.csv')
    prices = pd.read_csv('Datasets/sell_prices.csv')
    return calendar, sales, prices

# SBC Classification (ADI vs CV^2)
def classify_sbc(series):
    # series: non-zero demand intervals and values
    demand_dates = series[series > 0].index
    if len(demand_dates) <= 1:
        return 'Intermittent' # Default for extremely sparse
    
    # ADI: Average Demand Interval
    # For daily data, ADI = total_periods / non_zero_demand_periods
    adi = len(series) / len(demand_dates)
    
    # CV^2: Coefficient of Variation squared of non-zero demand
    non_zero_demand = series[series > 0]
    cv2 = (non_zero_demand.std() / non_zero_demand.mean())**2
    
    if adi < 1.32 and cv2 < 0.49:
        return 'Smooth'
    elif adi >= 1.32 and cv2 < 0.49:
        return 'Intermittent'
    elif adi < 1.32 and cv2 >= 0.49:
        return 'Erratic'
    else:
        return 'Lumpy'

# ABC/XYZ Analysis
def classify_abc_xyz(df_items):
    # df_items should have 'revenue' and 'cv' columns
    # ABC based on revenue (80/15/5 rule)
    df_items = df_items.sort_values('revenue', ascending=False)
    df_items['rev_cum_pct'] = df_items['revenue'].cumsum() / df_items['revenue'].sum()
    df_items['ABC'] = pd.cut(df_items['rev_cum_pct'], bins=[0, 0.8, 0.95, 1.0], labels=['A', 'B', 'C'], include_lowest=True)
    
    # XYZ based on demand variability (CV)
    # Common thresholds: X < 0.5, 0.5 <= Y < 1.0, Z >= 1.0
    df_items['XYZ'] = pd.cut(df_items['cv'], bins=[0, 0.5, 1.0, np.inf], labels=['X', 'Y', 'Z'], include_lowest=True)
    return df_items

# Decomposition and Stationarity
def analyze_series(series, name):
    # STL Decomposition
    res = STL(series, period=7).fit()
    
    # Stationarity
    adf_res = adfuller(series)
    kpss_res = kpss(series, regression='c')
    
    return {
        'trend': res.trend,
        'seasonal': res.seasonal,
        'resid': res.resid,
        'adf_p': adf_res[1],
        'kpss_p': kpss_res[1]
    }

if __name__ == "__main__":
    print("Loading data...")
    calendar, sales, prices = load_data()
    
    # Pre-process: merge with hierarchy
    hierarchy_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    
    # 1. Classification at Item-Store Level (30,490 series)
    # The prompt says 3,049 items, but let's see if it wants across stores.
    # "Decompose 3,049 Walmart product time series across 10 stores" 
    # Usually means the 3,049 items in the dataset.
    
    # Let's aggregate to item level (sum across stores) for the 3,049 series
    item_sales = sales.groupby('item_id')[d_cols].sum()
    
    print(f"Analyzing {len(item_sales)} item series...")
    
    # Classification and Stationarity for a few samples
    results = []
    stationarity_results = []
    
    # Sample one from each category for deep analysis
    sample_items = []
    
    for item_id, series in item_sales.iterrows():
        sbc = classify_sbc(series)
        cv = series.std() / series.mean() if series.mean() > 0 else 0
        revenue = series.sum() 
        results.append({'item_id': item_id, 'sbc': sbc, 'cv': cv, 'revenue': revenue})
        
        # Test stationarity for first 100 items to get a feel
        if len(stationarity_results) < 100:
            try:
                adf_p = adfuller(series)[1]
                kpss_p = kpss(series, regression='c')[1]
                stationarity_results.append({
                    'item_id': item_id,
                    'adf_p': adf_p,
                    'kpss_p': kpss_p,
                    'is_stationary': (adf_p < 0.05) and (kpss_p > 0.05)
                })
            except:
                pass

    df_results = pd.DataFrame(results)
    df_results = classify_abc_xyz(df_results)
    
    df_stat = pd.DataFrame(stationarity_results)
    
    # Merge results back to get category/dept info
    item_info = sales[['item_id', 'dept_id', 'cat_id']].drop_duplicates()
    df_results = df_results.merge(item_info, on='item_id', how='left')
    
    # Aggregations
    cat_sbc = df_results.groupby('cat_id')['sbc'].value_counts(normalize=True).unstack()
    dept_sbc = df_results.groupby('dept_id')['sbc'].value_counts(normalize=True).unstack()
    
    print("\nSBC Distribution by Category:")
    print(cat_sbc)
    
    # Save all
    df_results.to_csv('demand_classification_results.csv', index=False)
    df_stat.to_csv('stationarity_results.csv', index=False)
    cat_sbc.to_csv('category_sbc_distribution.csv')
    
    # Sample Decomposition for the notebook
    # We'll pick one 'Smooth' item for a nice decomposition plot
    smooth_sample = df_results[df_results['sbc'] == 'Smooth'].iloc[0]['item_id']
    series_sample = item_sales.loc[smooth_sample]
    res = STL(series_sample, period=7).fit()
    
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(series_sample)
    plt.title(f'Original Series: {smooth_sample}')
    plt.subplot(4, 1, 2)
    plt.plot(res.trend)
    plt.title('Trend')
    plt.subplot(4, 1, 3)
    plt.plot(res.seasonal)
    plt.title('Seasonality (7-day)')
    plt.subplot(4, 1, 4)
    plt.plot(res.resid)
    plt.title('Residual')
    plt.tight_layout()
    plt.savefig('sample_decomposition.png')
    print(f"\nSaved sample decomposition for {smooth_sample}")


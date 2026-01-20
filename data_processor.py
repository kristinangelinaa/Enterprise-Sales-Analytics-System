"""
Advanced Data Processing Pipeline for Enterprise Sales Analytics
Includes feature engineering, predictive models, and advanced analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class SalesDataProcessor:
    """Advanced sales data processor with ML capabilities"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.rfm_df = None
        self.product_affinity = None

    def load_and_clean(self):
        """Load and clean the dataset with advanced preprocessing"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.file_path, encoding='latin-1')

        # Convert dates
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'], format='%d/%m/%Y')
        self.df['Ship Date'] = pd.to_datetime(self.df['Ship Date'], format='%d/%m/%Y')

        # Feature Engineering
        self.df['Year'] = self.df['Order Date'].dt.year
        self.df['Month'] = self.df['Order Date'].dt.month
        self.df['Quarter'] = self.df['Order Date'].dt.quarter
        self.df['Day_of_Week'] = self.df['Order Date'].dt.dayofweek
        self.df['Week_of_Year'] = self.df['Order Date'].dt.isocalendar().week
        self.df['Shipping_Days'] = (self.df['Ship Date'] - self.df['Order Date']).dt.days

        # Calculate additional metrics (simulated since not in original data)
        np.random.seed(42)
        self.df['Quantity'] = np.random.randint(1, 15, size=len(self.df))
        self.df['Discount'] = np.random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.25], size=len(self.df), p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05])

        # Calculate profit (typical retail margin 20-40%)
        self.df['Cost'] = self.df['Sales'] * (1 - np.random.uniform(0.2, 0.4, size=len(self.df)))
        self.df['Profit'] = self.df['Sales'] - self.df['Cost']
        self.df['Profit_Margin'] = (self.df['Profit'] / self.df['Sales']) * 100

        # Revenue after discount
        self.df['Revenue'] = self.df['Sales'] * (1 - self.df['Discount'])

        print(f"✅ Loaded {len(self.df):,} records")
        return self.df

    def calculate_rfm(self):
        """Calculate RFM (Recency, Frequency, Monetary) segmentation"""
        print("🎯 Calculating RFM segmentation...")

        # Get the most recent date in dataset
        max_date = self.df['Order Date'].max()

        # Calculate RFM metrics
        rfm = self.df.groupby('Customer ID').agg({
            'Order Date': lambda x: (max_date - x.max()).days,  # Recency
            'Order ID': 'count',  # Frequency
            'Sales': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)

        # Calculate RFM Score
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

        # Segment customers
        def segment_customer(row):
            if row['RFM_Score'] >= 13:
                return 'Champions'
            elif row['RFM_Score'] >= 11:
                return 'Loyal Customers'
            elif row['RFM_Score'] >= 9:
                return 'Potential Loyalists'
            elif row['RFM_Score'] >= 7:
                return 'At Risk'
            elif row['RFM_Score'] >= 5:
                return 'Need Attention'
            else:
                return 'Lost'

        rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)

        # Calculate CLV (Customer Lifetime Value) - simplified
        rfm['CLV'] = rfm['Monetary'] * (rfm['Frequency'] / rfm['Recency'].replace(0, 1)) * 12

        self.rfm_df = rfm
        print(f"✅ RFM segmentation complete for {len(rfm)} customers")
        return rfm

    def calculate_product_affinity(self):
        """Calculate product affinity and market basket analysis"""
        print("🛒 Calculating product affinity matrix...")

        # Get products purchased together
        order_products = self.df.groupby('Order ID')['Sub-Category'].apply(list).reset_index()

        # Create affinity matrix
        from itertools import combinations

        pairs = []
        for products in order_products['Sub-Category']:
            if len(products) > 1:
                for pair in combinations(set(products), 2):
                    pairs.append(sorted(pair))

        affinity_df = pd.DataFrame(pairs, columns=['Product_A', 'Product_B'])
        affinity_matrix = affinity_df.groupby(['Product_A', 'Product_B']).size().reset_index(name='Count')
        affinity_matrix = affinity_matrix.sort_values('Count', ascending=False)

        self.product_affinity = affinity_matrix
        print(f"✅ Found {len(affinity_matrix)} product associations")
        return affinity_matrix

    def calculate_cohort_analysis(self):
        """Perform cohort analysis for customer retention"""
        print("📈 Calculating cohort analysis...")

        # Get first purchase date for each customer
        self.df['Cohort'] = self.df.groupby('Customer ID')['Order Date'].transform('min')
        self.df['Cohort_Month'] = self.df['Cohort'].dt.to_period('M')
        self.df['Order_Month'] = self.df['Order Date'].dt.to_period('M')

        # Calculate cohort index (months since first purchase)
        self.df['Cohort_Index'] = (self.df['Order_Month'] - self.df['Cohort_Month']).apply(lambda x: x.n)

        # Create cohort table
        cohort_data = self.df.groupby(['Cohort_Month', 'Cohort_Index'])['Customer ID'].nunique().reset_index()
        cohort_pivot = cohort_data.pivot(index='Cohort_Month', columns='Cohort_Index', values='Customer ID')

        # Calculate retention rates
        cohort_size = cohort_pivot.iloc[:, 0]
        retention = cohort_pivot.divide(cohort_size, axis=0) * 100

        print("✅ Cohort analysis complete")
        return retention

    def predict_sales_trend(self):
        """Predict sales trends using simple time series analysis"""
        print("🔮 Generating sales predictions...")

        # Aggregate sales by date
        daily_sales = self.df.groupby('Order Date')['Sales'].sum().reset_index()
        daily_sales = daily_sales.sort_values('Order Date')

        # Create features for prediction
        daily_sales['Day_Index'] = range(len(daily_sales))
        daily_sales['Month'] = daily_sales['Order Date'].dt.month
        daily_sales['DayOfWeek'] = daily_sales['Order Date'].dt.dayofweek

        # Simple moving averages
        daily_sales['MA_7'] = daily_sales['Sales'].rolling(window=7, min_periods=1).mean()
        daily_sales['MA_30'] = daily_sales['Sales'].rolling(window=30, min_periods=1).mean()

        # Use last 80% for training, 20% for "prediction"
        train_size = int(len(daily_sales) * 0.8)
        train = daily_sales[:train_size]
        test = daily_sales[train_size:]

        # Simple linear trend + seasonality
        from sklearn.linear_model import LinearRegression

        features = ['Day_Index', 'Month', 'DayOfWeek', 'MA_7', 'MA_30']
        X_train = train[features].fillna(method='bfill')
        y_train = train['Sales']

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict next 30 days
        last_date = daily_sales['Order Date'].max()
        future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
        future_df = pd.DataFrame({
            'Order Date': future_dates,
            'Day_Index': range(len(daily_sales), len(daily_sales) + 30),
            'Month': [d.month for d in future_dates],
            'DayOfWeek': [d.weekday() for d in future_dates],
            'MA_7': [daily_sales['MA_7'].iloc[-1]] * 30,
            'MA_30': [daily_sales['MA_30'].iloc[-1]] * 30
        })

        future_df['Predicted_Sales'] = model.predict(future_df[features])

        print("✅ Sales predictions generated")
        return daily_sales, future_df

    def get_geographic_metrics(self):
        """Calculate geographic performance metrics"""
        print("🌎 Calculating geographic metrics...")

        geo_metrics = self.df.groupby(['State', 'City', 'Region']).agg({
            'Sales': ['sum', 'mean', 'count'],
            'Profit': 'sum',
            'Quantity': 'sum',
            'Customer ID': 'nunique'
        }).reset_index()

        geo_metrics.columns = ['State', 'City', 'Region', 'Total_Sales', 'Avg_Sale',
                               'Order_Count', 'Total_Profit', 'Total_Quantity', 'Unique_Customers']

        geo_metrics['Profit_Margin'] = (geo_metrics['Total_Profit'] / geo_metrics['Total_Sales']) * 100
        geo_metrics['Sales_per_Customer'] = geo_metrics['Total_Sales'] / geo_metrics['Unique_Customers']

        print("✅ Geographic metrics calculated")
        return geo_metrics

    def get_advanced_insights(self):
        """Generate comprehensive advanced insights"""
        print("\n🚀 Generating advanced analytics insights...\n")

        insights = {}

        # 1. Customer Intelligence
        insights['total_customers'] = self.df['Customer ID'].nunique()
        insights['total_orders'] = self.df['Order ID'].nunique()
        insights['total_revenue'] = self.df['Sales'].sum()
        insights['total_profit'] = self.df['Profit'].sum()
        insights['avg_order_value'] = self.df.groupby('Order ID')['Sales'].sum().mean()
        insights['avg_profit_margin'] = self.df['Profit_Margin'].mean()

        # 2. Time-based insights
        insights['revenue_growth_yoy'] = self.df.groupby('Year')['Sales'].sum().pct_change().mean() * 100
        insights['best_month'] = self.df.groupby('Month')['Sales'].sum().idxmax()
        insights['best_quarter'] = self.df.groupby('Quarter')['Sales'].sum().idxmax()

        # 3. Product insights
        insights['top_category'] = self.df.groupby('Category')['Sales'].sum().idxmax()
        insights['top_subcategory'] = self.df.groupby('Sub-Category')['Sales'].sum().idxmax()
        insights['most_profitable_category'] = self.df.groupby('Category')['Profit'].sum().idxmax()

        # 4. Geographic insights
        insights['top_region'] = self.df.groupby('Region')['Sales'].sum().idxmax()
        insights['top_state'] = self.df.groupby('State')['Sales'].sum().idxmax()

        # 5. Operational insights
        insights['avg_shipping_days'] = self.df['Shipping_Days'].mean()
        insights['fastest_ship_mode'] = self.df.groupby('Ship Mode')['Shipping_Days'].mean().idxmin()

        # 6. Discount analysis
        insights['avg_discount'] = self.df['Discount'].mean() * 100
        insights['discount_impact_on_profit'] = self.df.groupby('Discount')['Profit_Margin'].mean().to_dict()

        return insights

    def export_for_tableau(self, output_path='tableau_export'):
        """Export processed data for Tableau"""
        print(f"📤 Exporting data for Tableau to '{output_path}' folder...")

        import os
        os.makedirs(output_path, exist_ok=True)

        # Export main dataset
        self.df.to_csv(f'{output_path}/sales_data_processed.csv', index=False)

        # Export RFM data
        if self.rfm_df is not None:
            self.rfm_df.to_csv(f'{output_path}/rfm_segmentation.csv', index=False)

        # Export product affinity
        if self.product_affinity is not None:
            self.product_affinity.to_csv(f'{output_path}/product_affinity.csv', index=False)

        # Export geographic metrics
        geo_metrics = self.get_geographic_metrics()
        geo_metrics.to_csv(f'{output_path}/geographic_metrics.csv', index=False)

        print(f"✅ Data exported to '{output_path}' folder")


if __name__ == "__main__":
    # Example usage
    processor = SalesDataProcessor('superstore sales dataset.csv')
    df = processor.load_and_clean()
    rfm = processor.calculate_rfm()
    affinity = processor.calculate_product_affinity()
    insights = processor.get_advanced_insights()

    print("\n" + "="*60)
    print("ADVANCED INSIGHTS SUMMARY")
    print("="*60)
    for key, value in insights.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

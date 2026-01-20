"""
Enterprise Sales Analytics Dashboard with 3D Visualizations
Built with Streamlit and Plotly for advanced interactive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_processor import SalesDataProcessor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enterprise Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and process data with caching"""
    processor = SalesDataProcessor('superstore sales dataset.csv')
    df = processor.load_and_clean()
    rfm = processor.calculate_rfm()
    affinity = processor.calculate_product_affinity()
    cohort = processor.calculate_cohort_analysis()
    daily_sales, predictions = processor.predict_sales_trend()
    geo_metrics = processor.get_geographic_metrics()
    insights = processor.get_advanced_insights()

    return df, rfm, affinity, cohort, daily_sales, predictions, geo_metrics, insights, processor


# Load data
with st.spinner('🚀 Loading enterprise analytics data...'):
    df, rfm, affinity, cohort, daily_sales, predictions, geo_metrics, insights, processor = load_data()

# Ultra-Compact Sidebar with CSS
st.sidebar.markdown("""
<style>
    .stSidebar > div:first-child {padding-top: 0.5rem;}
    .stSidebar .stMarkdown {margin-bottom: 0.3rem;}
    .stSidebar .element-container {margin-bottom: 0.3rem;}
    section[data-testid="stSidebar"] label {font-size: 0.85rem; font-weight: 500; margin-bottom: 0.2rem;}
    section[data-testid="stSidebar"] .stSelectbox {margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ Filters")

# Date range filter (always visible)
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()

date_range = st.sidebar.date_input(
    "📅 Date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region filter (compact expander with multiselect)
regions = sorted(df['Region'].unique().tolist())
with st.sidebar.expander("🌍 Region", expanded=False):
    select_all_regions = st.checkbox("Select All Regions", value=True, key="regions_all")
    if select_all_regions:
        selected_regions = regions
    else:
        selected_regions = st.multiselect(
            "Choose regions",
            options=regions,
            default=[],
            label_visibility="collapsed"
        )

# Category filter (compact expander with multiselect)
categories = sorted(df['Category'].unique().tolist())
with st.sidebar.expander("📦 Category", expanded=False):
    select_all_categories = st.checkbox("Select All Categories", value=True, key="categories_all")
    if select_all_categories:
        selected_categories = categories
    else:
        selected_categories = st.multiselect(
            "Choose categories",
            options=categories,
            default=[],
            label_visibility="collapsed"
        )

# Segment filter (compact expander with multiselect)
segments = sorted(df['Segment'].unique().tolist())
with st.sidebar.expander("👥 Segment", expanded=False):
    select_all_segments = st.checkbox("Select All Segments", value=True, key="segments_all")
    if select_all_segments:
        selected_segments = segments
    else:
        selected_segments = st.multiselect(
            "Choose segments",
            options=segments,
            default=[],
            label_visibility="collapsed"
        )

# Apply filters
filtered_df = df.copy()

# Handle date range filter safely
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['Order Date'].dt.date >= start_date) &
        (filtered_df['Order Date'].dt.date <= end_date)
    ]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    # Single date selected
    single_date = date_range[0]
    filtered_df = filtered_df[filtered_df['Order Date'].dt.date == single_date]

# Apply multiselect filters (empty list means show all)
if selected_regions and len(selected_regions) > 0:
    filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]

if selected_categories and len(selected_categories) > 0:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]

if selected_segments and len(selected_segments) > 0:
    filtered_df = filtered_df[filtered_df['Segment'].isin(selected_segments)]

# Safety check: ensure filtered_df is not empty
if len(filtered_df) == 0:
    st.warning("⚠️ No data available for the selected filters. Please adjust your date range or filter selections.")
    st.stop()

# Main dashboard
st.markdown("<br>", unsafe_allow_html=True)  # Add space at top
st.title("📊 Enterprise Sales Analytics Dashboard")
st.markdown("### Advanced Analytics with Predictive Insights & 3D Visualizations")
st.markdown("---")

# KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_revenue = filtered_df['Sales'].sum()
    # Calculate YoY from filtered data
    try:
        if 'Year' in filtered_df.columns and len(filtered_df['Year'].unique()) > 1:
            yearly_sales = filtered_df.groupby('Year')['Sales'].sum()
            if len(yearly_sales) >= 2:
                yoy_growth_filtered = yearly_sales.pct_change().mean() * 100
            else:
                yoy_growth_filtered = 0
        else:
            yoy_growth_filtered = insights['revenue_growth_yoy']
    except:
        yoy_growth_filtered = insights['revenue_growth_yoy']

    st.metric("💰 Total Revenue", f"${total_revenue:,.0f}",
              delta=f"{yoy_growth_filtered:.1f}% YoY")

with col2:
    total_profit = filtered_df['Profit'].sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    st.metric("💎 Total Profit", f"${total_profit:,.0f}",
              delta=f"{profit_margin:.1f}% margin")

with col3:
    total_orders = filtered_df['Order ID'].nunique()
    st.metric("📦 Total Orders", f"{total_orders:,}")

with col4:
    total_customers = filtered_df['Customer ID'].nunique()
    st.metric("👥 Unique Customers", f"{total_customers:,}")

with col5:
    avg_order_value = filtered_df.groupby('Order ID')['Sales'].sum().mean()
    st.metric("🎯 Avg Order Value", f"${avg_order_value:.2f}")

st.markdown("---")

# Add Growth Metrics Section
st.subheader("📊 Executive Growth Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Calculate MoM growth safely
    try:
        if len(filtered_df) > 0:
            current_month_sales = filtered_df[filtered_df['Order Date'].dt.to_period('M') == filtered_df['Order Date'].dt.to_period('M').max()]['Sales'].sum()
            prev_month_data = filtered_df[filtered_df['Order Date'].dt.to_period('M') == (filtered_df['Order Date'].dt.to_period('M').max() - 1)]
            prev_month_sales = prev_month_data['Sales'].sum() if len(prev_month_data) > 0 else 0
            mom_growth = ((current_month_sales - prev_month_sales) / prev_month_sales * 100) if prev_month_sales > 0 else 0
        else:
            current_month_sales = 0
            prev_month_sales = 0
            mom_growth = 0
    except:
        mom_growth = 0
        current_month_sales = 0
        prev_month_sales = 0

    st.metric("📈 MoM Growth", f"{mom_growth:.1f}%",
              delta=f"${current_month_sales - prev_month_sales:,.0f}")
    st.caption("Month-over-Month revenue change")

with col2:
    # YoY Growth - use the one calculated above
    st.metric("📅 YoY Growth", f"{yoy_growth_filtered:.1f}%")
    st.caption("Year-over-Year performance")

with col3:
    # QoQ Growth
    try:
        if len(filtered_df) > 0 and 'Quarter' in filtered_df.columns:
            current_q_sales = filtered_df[filtered_df['Quarter'] == filtered_df['Quarter'].max()]['Sales'].sum()
            prev_q_data = filtered_df[filtered_df['Quarter'] == (filtered_df['Quarter'].max() - 1)]
            prev_q_sales = prev_q_data['Sales'].sum() if len(prev_q_data) > 0 else 0
            qoq_growth = ((current_q_sales - prev_q_sales) / prev_q_sales * 100) if prev_q_sales > 0 else 0
        else:
            qoq_growth = 0
    except:
        qoq_growth = 0

    st.metric("📆 QoQ Growth", f"{qoq_growth:.1f}%")
    st.caption("Quarter-over-Quarter trend")

with col4:
    # Average profit margin
    try:
        if len(filtered_df) > 0 and 'Profit_Margin' in filtered_df.columns:
            avg_margin = filtered_df['Profit_Margin'].mean()
        else:
            avg_margin = 0
    except:
        avg_margin = 0

    st.metric("💰 Avg Profit Margin", f"{avg_margin:.1f}%")
    st.caption("Overall profitability")

st.markdown("---")

# Tabs for different sections
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 3D Analytics",
    "🎯 Customer Intelligence",
    "🛒 Product Analytics",
    "🌍 Geographic Insights",
    "🔮 Predictive Analytics",
    "📤 Tableau Export"
])

# TAB 0: OVERVIEW (Product-wise & Discount Impact)
with tab0:
    st.header("📊 Executive Overview")

    # Product-wise & Region-wise Performance
    st.subheader("🎯 Product-wise & Region-wise Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Products by Revenue**")
        top_products = filtered_df.groupby('Sub-Category')['Sales'].sum().nlargest(5).reset_index()
        top_products['Sales'] = top_products['Sales'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_products, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Regional Performance Summary**")
        regional_perf = filtered_df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit_Margin': 'mean'
        }).reset_index()
        regional_perf['Sales'] = regional_perf['Sales'].apply(lambda x: f"${x:,.0f}")
        regional_perf['Profit_Margin'] = regional_perf['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(regional_perf, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Discount Impact Analysis
    st.subheader("💸 Profit Margin & Discount Impact")

    with st.expander("ℹ️ How to Read This Chart"):
        st.markdown("""
        **What This Shows:**
        The relationship between discount levels and business performance

        **Two Metrics Displayed:**
        - **Blue Bars** = Total revenue at each discount level (left axis)
        - **Red Line** = Profit margin percentage (right axis)

        **What to Look For:**
        - **Optimal discount** = High revenue bar + High profit line
        - **Dangerous discount** = Low revenue bar + Low profit line (losing money and volume)
        - **Volume driver** = Very high revenue but declining profit (consider if worth it)

        **Business Decision:** Choose discounts that maximize both revenue AND maintain healthy margins (30%+)
        """)

    # Prepare discount data
    discount_viz = filtered_df.groupby('Discount').agg({
        'Sales': 'sum',
        'Profit_Margin': 'mean',
        'Order ID': 'count'
    }).reset_index()
    discount_viz['Discount_Label'] = discount_viz['Discount'].apply(lambda x: f"{x*100:.0f}%")

    # Create dual-axis chart
    fig_discount = make_subplots(specs=[[{"secondary_y": True}]])

    # Add revenue bars
    fig_discount.add_trace(
        go.Bar(
            x=discount_viz['Discount_Label'],
            y=discount_viz['Sales'],
            name='Total Revenue',
            marker_color='#3498db',
            text=discount_viz['Sales'].apply(lambda x: f"${x/1000:.0f}K"),
            textposition='outside'
        ),
        secondary_y=False
    )

    # Add profit margin line
    fig_discount.add_trace(
        go.Scatter(
            x=discount_viz['Discount_Label'],
            y=discount_viz['Profit_Margin'],
            name='Profit Margin %',
            mode='lines+markers+text',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            text=discount_viz['Profit_Margin'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center'
        ),
        secondary_y=True
    )

    # Update layout
    fig_discount.update_layout(
        title='Discount Level Impact: Revenue vs Profit Margin',
        xaxis_title='Discount Level',
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_discount.update_yaxes(title_text="<b>Total Revenue ($)</b>", secondary_y=False)
    fig_discount.update_yaxes(title_text="<b>Profit Margin (%)</b>", secondary_y=True, range=[0, max(discount_viz['Profit_Margin']) * 1.2])

    st.plotly_chart(fig_discount, use_container_width=True)

    # Strategic insights about discounting
    st.markdown("### 🎯 Discount Strategy Insights")

    col1, col2, col3 = st.columns(3)

    # Calculate discount impact metrics
    no_discount = discount_viz[discount_viz['Discount'] == 0].iloc[0] if len(discount_viz[discount_viz['Discount'] == 0]) > 0 else None
    with_discount = discount_viz[discount_viz['Discount'] > 0]

    if with_discount.empty:
        avg_discount_revenue = 0
        avg_discount_margin = 0
    else:
        avg_discount_revenue = with_discount['Sales'].mean()
        avg_discount_margin = with_discount['Profit_Margin'].mean()

    with col1:
        if no_discount is not None:
            st.success(f"""
            **💎 No Discount Performance**

            - Revenue: ${no_discount['Sales']:,.0f}
            - Margin: {no_discount['Profit_Margin']:.1f}%
            - Orders: {no_discount['Order ID']:,}

            *Highest profitability*
            """)
        else:
            st.success("**💎 No Discount Data**\n\nNo orders without discounts")

    with col2:
        # Find most popular discount
        popular_discount = discount_viz.nlargest(1, 'Order ID').iloc[0]
        st.info(f"""
        **🎯 Most Used Discount**

        **{popular_discount['Discount_Label']}**
        - Orders: {popular_discount['Order ID']:,}
        - Revenue: ${popular_discount['Sales']:,.0f}
        - Margin: {popular_discount['Profit_Margin']:.1f}%

        *Customer favorite*
        """)

    with col3:
        # Calculate discount effectiveness
        if no_discount is not None and not with_discount.empty:
            revenue_lift = ((avg_discount_revenue - no_discount['Sales']) / no_discount['Sales']) * 100 if no_discount['Sales'] > 0 else 0
            margin_drop = no_discount['Profit_Margin'] - avg_discount_margin

            if revenue_lift > 0:
                st.warning(f"""
                **📊 Discount Impact**

                **Average with discounts:**
                - Revenue: {revenue_lift:+.1f}% vs no discount
                - Margin drop: {margin_drop:.1f}%

                *Trade-off analysis*
                """)
            else:
                st.warning(f"""
                **📊 Discount Impact**

                **Average with discounts:**
                - Revenue: {revenue_lift:.1f}% vs no discount
                - Margin drop: {margin_drop:.1f}%

                *Consider reducing discounts*
                """)
        else:
            st.warning("**📊 Discount Impact**\n\nInsufficient data for comparison")

    st.caption("💡 **Strategy:** Blue bars show revenue impact, red line shows profitability. The optimal discount level balances both metrics.")

# TAB 1: 3D ANALYTICS
with tab1:
    st.header("🎨 3D Visualizations")

    st.info("""
    **📖 How to Read 3D Charts:**
    - **Click & Drag** to rotate and view from different angles
    - **Scroll** to zoom in/out
    - **Hover** over points to see detailed values
    - **Larger bubbles** = Higher sales values
    - **Color intensity** = Sales performance (darker = higher)
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("3D Sales Cube: Time × Region × Category")

        with st.expander("ℹ️ What does this show?"):
            st.markdown("""
            **Purpose:** See how sales perform across three dimensions simultaneously

            **How to interpret:**
            - **Bubble size** = Total sales amount
            - **Position** = Combination of Year, Region, and Product Category
            - **Color** = Sales intensity (bright yellow = highest, dark purple = lowest)

            **Business Use:**
            - Identify which category performs best in which region over time
            - Spot seasonal trends across different product lines
            - Compare performance across multiple years at a glance
            """)

        # Aggregate data for 3D visualization
        sales_cube = filtered_df.groupby(['Year', 'Region', 'Category'])['Sales'].sum().reset_index()

        fig_3d_scatter = px.scatter_3d(
            sales_cube,
            x='Year',
            y='Region',
            z='Category',
            size='Sales',
            color='Sales',
            color_continuous_scale='Viridis',
            title='3D Sales Distribution',
            hover_data=['Sales'],
            size_max=50
        )

        fig_3d_scatter.update_layout(
            scene=dict(
                xaxis_title='Year',
                yaxis_title='Region',
                zaxis_title='Category',
                bgcolor='rgba(240,240,240,0.9)'
            ),
            height=500
        )

        st.plotly_chart(fig_3d_scatter, use_container_width=True)
        st.caption("💡 **Insight:** Larger, brighter bubbles indicate your star performers")

    with col2:
        st.subheader("3D Geographic Sales Distribution")

        with st.expander("ℹ️ What does this show?"):
            st.markdown("""
            **Purpose:** Visualize sales performance across geographic locations in 3D space

            **How to interpret:**
            - **X & Y axes** = Geographic position (Longitude & Latitude)
            - **Height (Z-axis)** = Sales amount (taller = more sales)
            - **Bubble size** = Market size
            - **Color** = Performance level (red = highest, purple = lowest)

            **Business Use:**
            - Identify high-performing geographic markets
            - Spot underperforming regions that need attention
            - Plan sales territory assignments
            - Decide warehouse or store locations
            """)

        # Create synthetic lat/lon for demo (in production, use geocoding)
        geo_3d = filtered_df.groupby(['State', 'Region'])['Sales'].sum().reset_index()

        # Approximate coordinates for US states (simplified)
        state_coords = {
            'California': (36.7783, -119.4179),
            'Texas': (31.9686, -99.9018),
            'New York': (40.7128, -74.0060),
            'Florida': (27.9944, -81.7603),
            'Illinois': (40.6331, -89.3985),
            # Add more as needed
        }

        geo_3d['lat'] = geo_3d['State'].map(lambda x: state_coords.get(x, (37.0, -95.0))[0])
        geo_3d['lon'] = geo_3d['State'].map(lambda x: state_coords.get(x, (37.0, -95.0))[1])

        fig_geo_3d = go.Figure(data=[go.Scatter3d(
            x=geo_3d['lon'],
            y=geo_3d['lat'],
            z=geo_3d['Sales'],
            mode='markers',
            marker=dict(
                size=geo_3d['Sales'] / 10000,
                color=geo_3d['Sales'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Sales")
            ),
            text=geo_3d['State'],
            hovertemplate='<b>%{text}</b><br>Sales: $%{z:,.0f}<extra></extra>'
        )])

        fig_geo_3d.update_layout(
            title='3D Geographic Sales Map',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Sales ($)',
                bgcolor='rgba(240,240,240,0.9)'
            ),
            height=500
        )

        st.plotly_chart(fig_geo_3d, use_container_width=True)
        st.caption("💡 **Insight:** Taller peaks show your strongest markets geographically")

    st.markdown("---")

    # 3D Surface Plot: Sales Heatmap over Time
    st.subheader("3D Surface Plot: Daily Sales Trends by Category")

    with st.expander("ℹ️ What does this show?"):
        st.markdown("""
        **Purpose:** See sales trends over time for each product category as a continuous surface

        **How to interpret:**
        - **X-axis** = Timeline (dates)
        - **Y-axis** = Product categories
        - **Height & Color** = Sales amount
        - **Peaks** = High sales periods
        - **Valleys** = Low sales periods
        - **Colors** = Red (hot) = high sales, Blue (cool) = low sales

        **Business Use:**
        - Identify seasonal patterns for each category
        - Spot which categories have consistent performance vs volatile ones
        - Plan inventory based on category-specific trends
        - Time promotions when certain categories naturally dip
        """)

    # Prepare data for surface plot
    daily_category = filtered_df.groupby(['Order Date', 'Category'])['Sales'].sum().reset_index()
    daily_category_pivot = daily_category.pivot(index='Order Date', columns='Category', values='Sales').fillna(0)

    # Create surface plot
    fig_surface = go.Figure(data=[go.Surface(
        z=daily_category_pivot.values.T,
        x=daily_category_pivot.index,
        y=daily_category_pivot.columns,
        colorscale='Turbo',
        hovertemplate='Date: %{x}<br>Category: %{y}<br>Sales: $%{z:,.0f}<extra></extra>'
    )])

    fig_surface.update_layout(
        title='3D Sales Surface: Time × Category',
        scene=dict(
            xaxis_title='Date',
            yaxis_title='Category',
            zaxis_title='Sales ($)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600
    )

    st.plotly_chart(fig_surface, use_container_width=True)
    st.caption("💡 **Insight:** Mountains (peaks) show strong sales periods, valleys indicate slower times")

# TAB 2: CUSTOMER INTELLIGENCE
with tab2:
    st.header("🎯 Customer Intelligence & RFM Analysis")

    # Calculate RFM on filtered data
    if len(filtered_df) > 0:
        max_date_filtered = filtered_df['Order Date'].max()

        rfm_filtered = filtered_df.groupby('Customer ID').agg({
            'Order Date': lambda x: (max_date_filtered - x.max()).days,
            'Order ID': 'count',
            'Sales': 'sum'
        }).reset_index()

        rfm_filtered.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

        # Calculate RFM scores
        rfm_filtered['R_Score'] = pd.qcut(rfm_filtered['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm_filtered['F_Score'] = pd.qcut(rfm_filtered['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm_filtered['M_Score'] = pd.qcut(rfm_filtered['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        rfm_filtered['R_Score'] = rfm_filtered['R_Score'].astype(int)
        rfm_filtered['F_Score'] = rfm_filtered['F_Score'].astype(int)
        rfm_filtered['M_Score'] = rfm_filtered['M_Score'].astype(int)

        rfm_filtered['RFM_Score'] = rfm_filtered['R_Score'] + rfm_filtered['F_Score'] + rfm_filtered['M_Score']

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

        rfm_filtered['Customer_Segment'] = rfm_filtered.apply(segment_customer, axis=1)
        rfm_filtered['CLV'] = rfm_filtered['Monetary'] * (rfm_filtered['Frequency'] / rfm_filtered['Recency'].replace(0, 1)) * 12
    else:
        rfm_filtered = rfm  # Fallback to preloaded if no data

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", f"{rfm_filtered['Customer ID'].nunique():,}")
    with col2:
        avg_clv = rfm_filtered['CLV'].mean()
        st.metric("Avg Customer Lifetime Value", f"${avg_clv:,.2f}")
    with col3:
        champions = len(rfm_filtered[rfm_filtered['Customer_Segment'] == 'Champions'])
        st.metric("Champion Customers", f"{champions:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segmentation (RFM)")

        segment_counts = rfm_filtered['Customer_Segment'].value_counts()

        fig_segment = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        st.plotly_chart(fig_segment, use_container_width=True)

    with col2:
        st.subheader("RFM Score Distribution")

        with st.expander("ℹ️ What is RFM Analysis?"):
            st.markdown("""
            **RFM = Recency, Frequency, Monetary**

            **How to interpret:**
            - **X-axis (Recency)** = Days since last purchase (lower = more recent = better)
            - **Y-axis (Frequency)** = Number of orders (higher = better)
            - **Z-axis (Monetary)** = Total money spent (higher = better)
            - **Bubble size** = Customer Lifetime Value
            - **Colors** = Customer segments

            **Customer Segments Explained:**
            - **Champions** (Best) = Recent buyers, frequent, high spend
            - **Loyal** = Good customers worth keeping engaged
            - **At Risk** = Were good, now declining - need attention!
            - **Lost** = Haven't purchased in a long time

            **Action:** Focus marketing on Champions and rescue At Risk customers
            """)

        fig_rfm = px.scatter_3d(
            rfm_filtered,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Customer_Segment',
            size='CLV',
            title='3D RFM Analysis',
            hover_data=['RFM_Score'],
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig_rfm.update_layout(height=500)
        st.plotly_chart(fig_rfm, use_container_width=True)
        st.caption("💡 **Insight:** Customers in the top-right corner are your VIPs")

    st.markdown("---")

    # Top customers table
    st.subheader("🏆 Top 20 Customers by CLV")

    top_customers = rfm_filtered.nlargest(20, 'CLV')[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'CLV', 'Customer_Segment']]
    top_customers['CLV'] = top_customers['CLV'].apply(lambda x: f"${x:,.2f}")
    top_customers['Monetary'] = top_customers['Monetary'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(top_customers, use_container_width=True)

# TAB 3: PRODUCT ANALYTICS
with tab3:
    st.header("🛒 Product Analytics & Market Basket Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Product Affinities")
        st.markdown("*Products frequently bought together*")

        top_affinity = affinity.head(10)

        fig_affinity = px.bar(
            top_affinity,
            x='Count',
            y=[f"{row['Product_A']} + {row['Product_B']}" for _, row in top_affinity.iterrows()],
            orientation='h',
            title='Most Common Product Combinations',
            color='Count',
            color_continuous_scale='Blues'
        )

        fig_affinity.update_layout(yaxis_title='', xaxis_title='Co-occurrence Count')
        st.plotly_chart(fig_affinity, use_container_width=True)

    with col2:
        st.subheader("Category Performance Matrix")

        category_metrics = filtered_df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Order ID': 'nunique'
        }).reset_index()

        fig_category = px.scatter(
            category_metrics,
            x='Sales',
            y='Profit',
            size='Quantity',
            color='Category',
            title='Sales vs Profit by Category',
            hover_data=['Order ID'],
            size_max=60
        )

        st.plotly_chart(fig_category, use_container_width=True)

    st.markdown("---")

    # Sub-category analysis
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Sub-Category Performance Breakdown")

    with col2:
        with st.expander("💡 Quick Guide"):
            st.caption("""
            **Box Size** = Sales
            **Color**: 🟢 Green = High profit | 🔴 Red = Low profit

            **Strategy:**
            Large+Green = Stars
            Large+Red = Fix margins
            Small+Green = Grow
            Small+Red = Review
            """)

    subcategory_data = filtered_df.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean'
    }).reset_index()

    # Add formatted labels for display
    subcategory_data['Label'] = subcategory_data.apply(
        lambda x: f"{x['Sub-Category']}<br>${x['Sales']:,.0f}<br>{x['Profit_Margin']:.1f}% margin",
        axis=1
    )

    fig_treemap = px.treemap(
        subcategory_data,
        path=['Category', 'Sub-Category'],
        values='Sales',
        color='Profit_Margin',
        color_continuous_scale='RdYlGn',
        title='Product Hierarchy: Sales & Profitability',
        hover_data={
            'Sales': ':$,.0f',
            'Profit': ':$,.0f',
            'Profit_Margin': ':.1f%'
        }
    )

    fig_treemap.update_traces(
        textposition='middle center',
        textfont_size=12,
        marker=dict(line=dict(width=2, color='white'))
    )

    fig_treemap.update_layout(
        height=500,
        margin=dict(t=40, b=0, l=0, r=0),
        coloraxis_colorbar=dict(
            title="Profit<br>Margin %",
            ticksuffix="%"
        )
    )

    st.plotly_chart(fig_treemap, use_container_width=True)

    # Reduce gap between chart and insights
    st.markdown("<style>div.block-container{padding-top: 0rem;}</style>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Find best performer
        best_product = subcategory_data.nlargest(1, 'Sales').iloc[0]
        st.success(f"""**🏆 Top Revenue Generator**\n\n**{best_product['Sub-Category']}**\n• Sales: ${best_product['Sales']:,.0f}\n• Profit Margin: {best_product['Profit_Margin']:.1f}%""")

    with col2:
        # Find most profitable
        most_profitable = subcategory_data.nlargest(1, 'Profit_Margin').iloc[0]
        st.info(f"""**💎 Most Profitable**\n\n**{most_profitable['Sub-Category']}**\n• Profit Margin: {most_profitable['Profit_Margin']:.1f}%\n• Sales: ${most_profitable['Sales']:,.0f}""")

    with col3:
        # Find needs attention
        needs_attention = subcategory_data.nsmallest(1, 'Profit_Margin').iloc[0]
        st.warning(f"""**⚠️ Needs Attention**\n\n**{needs_attention['Sub-Category']}**\n• Profit Margin: {needs_attention['Profit_Margin']:.1f}%\n• Consider price increase or cost reduction""")

    st.caption("💡 **Strategy Tip:** Big green boxes are your champions. Big red boxes need margin improvement through better pricing or lower costs.")

# TAB 4: GEOGRAPHIC INSIGHTS
with tab4:
    st.header("🌍 Geographic Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regional Sales Distribution")

        regional_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()

        fig_region = px.bar(
            regional_sales,
            x='Region',
            y='Sales',
            title='Sales by Region',
            color='Sales',
            color_continuous_scale='Viridis'
        )

        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        st.subheader("Top 10 States by Revenue")

        state_sales = filtered_df.groupby('State')['Sales'].sum().nlargest(10).reset_index()

        fig_state = px.bar(
            state_sales,
            x='Sales',
            y='State',
            orientation='h',
            title='Top Performing States',
            color='Sales',
            color_continuous_scale='Plasma'
        )

        st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("---")

    # Regional Performance Map with Better Visualization
    st.subheader("Regional Sales Performance Map")

    with st.expander("ℹ️ What This Map Shows"):
        st.markdown("""
        **Purpose:** See at a glance where your sales are strongest and which states drive each region

        **How to Read the Map:**
        - **Bubble size** = Total sales revenue (bigger bubble = more sales)
        - **Color darkness** = Profit margin (darker purple = better profitability)
        - Each bubble = One of your 4 regions (East, West, Central, South)

        **Understanding Results:**
        - **Big + Dark Purple** = Star region (high sales + high profit) → Invest more!
        - **Big + Light Purple** = Volume region (high sales but lower profit) → Improve efficiency
        - **Small + Dark Purple** = Profitable niche (low sales but good margins) → Grow market share
        - **Small + Light Purple** = Weak region (low sales + low profit) → Need strategic review

        **Regional Breakdown:**
        - **East Region**: NY, Pennsylvania, Connecticut, etc.
        - **West Region**: California, Washington, Oregon, etc.
        - **Central Region**: Illinois, Texas, Wisconsin, etc.
        - **South Region**: Florida, Georgia, North Carolina, etc.

        Each region's performance is driven by specific states shown in the insights below.
        """)

    # Get top states per region for insights
    state_by_region = filtered_df.groupby(['Region', 'State'])['Sales'].sum().reset_index()
    top_states_per_region = state_by_region.sort_values(['Region', 'Sales'], ascending=[True, False]).groupby('Region').head(3)

    # State name to abbreviation mapping
    state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
        'District of Columbia': 'DC'
    }

    # Create state-level data for choropleth map
    state_data = filtered_df.groupby(['State']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean',
        'Region': 'first'
    }).reset_index()

    # Add state abbreviations
    state_data['State_Code'] = state_data['State'].map(state_abbrev)

    # Map regions to numeric values for color coding
    region_colors = {'Central': 1, 'East': 2, 'South': 3, 'West': 4}
    state_data['Region_Num'] = state_data['Region'].map(region_colors)

    # Create custom color scale for regions
    # Central=Purple, East=Blue, South=Orange, West=Green
    custom_colorscale = [
        [0, '#9b59b6'],      # Central - Purple
        [0.25, '#9b59b6'],
        [0.25, '#3498db'],   # East - Blue
        [0.5, '#3498db'],
        [0.5, '#e67e22'],    # South - Orange
        [0.75, '#e67e22'],
        [0.75, '#27ae60'],   # West - Green
        [1, '#27ae60']
    ]

    # Create choropleth map (filled states by region color)
    fig_map = go.Figure(data=go.Choropleth(
        locations=state_data['State_Code'],
        z=state_data['Region_Num'],
        locationmode='USA-states',
        colorscale=custom_colorscale,
        text=state_data['State'],
        customdata=state_data[['Region', 'Sales', 'Profit_Margin']],
        hovertemplate='<b>%{text}</b><br>Region: %{customdata[0]}<br>Sales: $%{customdata[1]:,.0f}<br>Profit Margin: %{customdata[2]:.1f}%<extra></extra>',
        showscale=False,
        marker_line_color='white',
        marker_line_width=2
    ))

    fig_map.update_layout(
        title='Sales Performance by Region (Color-Coded Map)',
        geo=dict(
            scope='usa',
            projection=go.layout.geo.Projection(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            bgcolor='rgba(243,243,243,1)'
        ),
        height=500
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Add regional insights with state breakdown
    st.markdown("### 🎯 Regional Performance Breakdown")
    st.caption("See which states drive each region's performance")

    # Create regional summary data for cards
    regional_map_data = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean',
        'Order ID': 'nunique'
    }).reset_index()

    # Create 4 columns for 4 regions
    regions = sorted(regional_map_data['Region'].unique())
    cols = st.columns(4)

    # Define region colors matching the map
    region_color_map = {
        'Central': ('#9b59b6', '#f4ecf7'),  # Purple
        'East': ('#3498db', '#ebf5fb'),      # Blue
        'South': ('#e67e22', '#fef5e7'),     # Orange
        'West': ('#27ae60', '#eafaf1')       # Green
    }

    for idx, region in enumerate(regions):
        with cols[idx]:
            region_data = regional_map_data[regional_map_data['Region'] == region].iloc[0]
            top_states = top_states_per_region[top_states_per_region['Region'] == region].nlargest(3, 'Sales')

            # Get colors for this region
            border_color, bg_color = region_color_map.get(region, ('#3498db', '#ebf5fb'))

            # Determine if this is the top region
            sales_rank = regional_map_data['Sales'].rank(ascending=False)[regional_map_data['Region'] == region].iloc[0]
            trophy = ' 🏆' if sales_rank == 1 else ''

            # Create custom colored card using HTML
            st.markdown(f"""
            <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; border-left: 5px solid {border_color};'>
                <h4 style='margin: 0; color: {border_color};'>{region}{trophy}</h4>
                <p style='margin: 10px 0 5px 0;'><strong>Sales:</strong> ${region_data['Sales']:,.0f}</p>
                <p style='margin: 5px 0;'><strong>Margin:</strong> {region_data['Profit_Margin']:.1f}%</p>
                <p style='margin: 10px 0 5px 0; font-weight: 600;'>Top States:</p>
            </div>
            """, unsafe_allow_html=True)

            # List top 3 states
            for _, state_row in top_states.iterrows():
                state_pct = (state_row['Sales'] / region_data['Sales']) * 100
                st.caption(f"• {state_row['State']}: ${state_row['Sales']:,.0f} ({state_pct:.1f}%)")

    st.markdown("---")

    st.caption("💡 **Strategy Insight:** Focus on the largest bubbles (highest sales regions) and their top contributing states. These are your primary markets for expansion and investment.")

    st.markdown("---")

    # City-level metrics
    st.subheader("Top 15 Cities Performance")

    with st.expander("ℹ️ Understanding City Performance"):
        st.markdown("""
        **What This Shows:**
        Detailed breakdown of your top-performing cities

        **Columns Explained:**
        - **Total_Sales** = Revenue generated in this city
        - **Total_Profit** = Profit earned in this city
        - **Profit_Margin** = Profitability percentage
        - **Unique_Customers** = Number of different customers
        - **Sales_per_Customer** = Average revenue per customer

        **Use This To:**
        - Identify which cities to prioritize for expansion
        - Spot cities with low sales per customer (upsell opportunity)
        - Find cities with high profit margins to replicate strategy
        """)

    # Calculate city metrics from filtered data
    city_metrics_filtered = filtered_df.groupby(['State', 'City', 'Region']).agg({
        'Sales': ['sum', 'mean', 'count'],
        'Profit': 'sum',
        'Quantity': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()

    city_metrics_filtered.columns = ['State', 'City', 'Region', 'Total_Sales', 'Avg_Sale',
                           'Order_Count', 'Total_Profit', 'Total_Quantity', 'Unique_Customers']

    city_metrics_filtered['Profit_Margin'] = (city_metrics_filtered['Total_Profit'] / city_metrics_filtered['Total_Sales']) * 100
    city_metrics_filtered['Sales_per_Customer'] = city_metrics_filtered['Total_Sales'] / city_metrics_filtered['Unique_Customers']

    city_metrics = city_metrics_filtered.nlargest(15, 'Total_Sales')

    # Format the dataframe for display
    city_display = city_metrics[['City', 'State', 'Region', 'Total_Sales', 'Total_Profit',
                                   'Profit_Margin', 'Unique_Customers', 'Sales_per_Customer']].copy()

    city_display['Total_Sales'] = city_display['Total_Sales'].apply(lambda x: f"${x:,.0f}")
    city_display['Total_Profit'] = city_display['Total_Profit'].apply(lambda x: f"${x:,.0f}")
    city_display['Profit_Margin'] = city_display['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
    city_display['Sales_per_Customer'] = city_display['Sales_per_Customer'].apply(lambda x: f"${x:,.0f}")

    st.dataframe(city_display, use_container_width=True, hide_index=True)

    st.caption("💡 **Tip:** Cities with high sales but low sales-per-customer have potential for upselling and customer engagement programs.")

# TAB 5: PREDICTIVE ANALYTICS
with tab5:
    st.header("🔮 Predictive Analytics & Forecasting")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Sales Forecast - Next 30 Days")

    with col2:
        with st.expander("💡 Chart Guide"):
            st.caption("""
            **Blue** = Past sales
            **Green** = Trend line
            **Red** = Forecast
            **Gray line** = Today

            📈 Up = Growth
            📉 Down = Slow
            """)

    # Generate forecast based on filtered data
    filtered_daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index().sort_values('Order Date')
    filtered_daily_sales['MA_7'] = filtered_daily_sales['Sales'].rolling(window=7, min_periods=1).mean()
    filtered_daily_sales['MA_30'] = filtered_daily_sales['Sales'].rolling(window=30, min_periods=1).mean()

    # Aggregate to weekly for cleaner visualization
    daily_sales_copy = filtered_daily_sales.copy()
    daily_sales_copy['Week'] = daily_sales_copy['Order Date'].dt.to_period('W').dt.start_time

    weekly_sales = daily_sales_copy.groupby('Week').agg({
        'Sales': 'sum',
        'MA_30': 'mean'
    }).reset_index()

    # Create cleaner forecast chart
    fig_forecast = go.Figure()

    # Historical sales as area (less noisy)
    fig_forecast.add_trace(go.Scatter(
        x=weekly_sales['Week'],
        y=weekly_sales['Sales'],
        mode='lines',
        name='Weekly Sales',
        fill='tozeroy',
        line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    # 30-day MA trend line (clear signal)
    fig_forecast.add_trace(go.Scatter(
        x=filtered_daily_sales['Order Date'],
        y=filtered_daily_sales['MA_30'],
        mode='lines',
        name='Trend (30-Day Avg)',
        line=dict(color='#2ecc71', width=3)
    ))

    # Add vertical line for "today"
    last_date = filtered_daily_sales['Order Date'].max()
    fig_forecast.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="solid")
    )

    fig_forecast.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        yshift=10
    )

    # Forecast (prominent and clear)
    fig_forecast.add_trace(go.Scatter(
        x=predictions['Order Date'],
        y=predictions['Predicted_Sales'],
        mode='lines+markers',
        name='30-Day Forecast',
        line=dict(color='#e74c3c', width=4, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))

    # Add shaded forecast region
    forecast_dates = predictions['Order Date'].tolist()
    forecast_upper = (predictions['Predicted_Sales'] * 1.15).tolist()  # +15% confidence
    forecast_lower = (predictions['Predicted_Sales'] * 0.85).tolist()  # -15% confidence

    fig_forecast.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=forecast_upper + forecast_lower[::-1],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='Confidence Range (±15%)',
        hoverinfo='skip'
    ))

    fig_forecast.update_layout(
        title='Sales Trend & 30-Day Forecast',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast insights
    col1, col2, col3 = st.columns(3)

    avg_historical = daily_sales['MA_30'].iloc[-1]
    avg_forecast = predictions['Predicted_Sales'].mean()
    forecast_change = ((avg_forecast - avg_historical) / avg_historical) * 100

    total_forecast = predictions['Predicted_Sales'].sum()

    with col1:
        if forecast_change > 0:
            st.success(f"""
            **📈 Forecast Trend**

            **+{forecast_change:.1f}% Growth Expected**

            Next 30 days predicted to be stronger than current trend.

            *Plan for increased demand*
            """)
        else:
            st.warning(f"""
            **📉 Forecast Trend**

            **{forecast_change:.1f}% Expected**

            Next 30 days may be slower than current trend.

            *Consider promotions*
            """)

    with col2:
        st.info(f"""
        **💰 Predicted Revenue**

        **${total_forecast:,.0f}**

        Total expected sales over next 30 days.

        *Based on historical patterns*
        """)

    with col3:
        avg_daily_forecast = total_forecast / 30
        st.metric(
            "📊 Daily Average (Forecast)",
            f"${avg_daily_forecast:,.0f}",
            delta=f"{forecast_change:.1f}% vs trend"
        )

    st.caption("💡 **Insight:** Green line shows your overall trend. Red forecast continues this pattern. Shaded area shows confidence range (±15%).")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Seasonality Analysis")

        monthly_avg = filtered_df.groupby('Month')['Sales'].mean().reset_index()
        monthly_avg['Month_Name'] = monthly_avg['Month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })

        fig_seasonality = px.line(
            monthly_avg,
            x='Month_Name',
            y='Sales',
            title='Average Sales by Month',
            markers=True
        )

        st.plotly_chart(fig_seasonality, use_container_width=True)

    with col2:
        st.subheader("Day of Week Performance")

        dow_sales = filtered_df.groupby('Day_of_Week')['Sales'].mean().reset_index()
        dow_sales['Day'] = dow_sales['Day_of_Week'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
        })

        fig_dow = px.bar(
            dow_sales,
            x='Day',
            y='Sales',
            title='Average Sales by Day of Week',
            color='Sales',
            color_continuous_scale='Blues'
        )

        st.plotly_chart(fig_dow, use_container_width=True)

# TAB 6: TABLEAU EXPORT
with tab6:
    st.header("📊 Tableau Integration & Data Export")

    st.markdown("""
    ### Export Options for Tableau Desktop

    This section allows you to export processed data for advanced visualization in Tableau.
    The exported files include:

    - **Sales Data (Processed)**: Main dataset with all calculated fields
    - **RFM Segmentation**: Customer intelligence data
    - **Product Affinity**: Market basket analysis results
    - **Geographic Metrics**: Location-based performance data
    """)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("📥 Export All Data for Tableau", type="primary", use_container_width=True):
            with st.spinner('Exporting data...'):
                processor.export_for_tableau('tableau_export')
                st.success("✅ Data exported successfully to 'tableau_export' folder!")

                st.balloons()

                st.markdown("""
                #### Next Steps:
                1. Navigate to the `tableau_export` folder
                2. Open Tableau Desktop
                3. Connect to CSV files or create a Hyper extract
                4. Build your custom visualizations
                """)

    st.markdown("---")

    # Show preview of exports
    st.subheader("📋 Data Preview")

    export_choice = st.selectbox(
        "Select dataset to preview:",
        ["Main Sales Data", "RFM Segmentation", "Product Affinity", "Geographic Metrics"]
    )

    if export_choice == "Main Sales Data":
        st.dataframe(filtered_df.head(100), use_container_width=True)
    elif export_choice == "RFM Segmentation":
        st.dataframe(rfm.head(100), use_container_width=True)
    elif export_choice == "Product Affinity":
        st.dataframe(affinity.head(100), use_container_width=True)
    elif export_choice == "Geographic Metrics":
        st.dataframe(geo_metrics.head(100), use_container_width=True)

    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Current View as CSV",
        data=csv,
        file_name='sales_data_filtered.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Enterprise Sales Analytics Dashboard</b> | Built with Streamlit & Plotly | Advanced Analytics & 3D Visualizations</p>
</div>
""", unsafe_allow_html=True)

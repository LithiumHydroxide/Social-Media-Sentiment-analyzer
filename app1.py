import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment & Anti-Narrative Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved text visibility
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card ul {
        color: #34495e;
        margin: 0;
        padding-left: 1.2rem;
    }
    .metric-card li {
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .alert-high { 
        border-left-color: #e74c3c;
        background: #fdf2f2;
    }
    .alert-high h4 {
        color: #c0392b;
    }
    .alert-medium { 
        border-left-color: #f39c12;
        background: #fef9e7;
    }
    .alert-medium h4 {
        color: #d68910;
    }
    .alert-low { 
        border-left-color: #27ae60;
        background: #eafaf1;
    }
    .alert-low h4 {
        color: #229954;
    }
    /* Dark theme adjustments */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #2c3e50;
            color: #ecf0f1;
        }
        .metric-card h4 {
            color: #ecf0f1;
        }
        .metric-card ul {
            color: #bdc3c7;
        }
        .alert-high {
            background: #4a2c2a;
            color: #ecf0f1;
        }
        .alert-medium {
            background: #4a3c2a;
            color: #ecf0f1;
        }
        .alert-low {
            background: #2a4a35;
            color: #ecf0f1;
        }
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data
@st.cache_data
def generate_sample_data():
    """Generate realistic sample data for the demo"""
    np.random.seed(42)
    
    # Sources and platforms
    sources = ['Twitter/X', 'Facebook', 'Reddit', 'LinkedIn', 'News Sites', 'Blogs', 'Forums']
    countries = ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'Brazil']
    topics = ['Economic Policy', 'Climate Change', 'Technology', 'Healthcare', 'Education', 'Immigration']
    
    # Generate headline data
    headlines_data = []
    for i in range(500):
        headline_data = {
            'id': i,
            'headline': f"Sample headline {i+1} about {random.choice(topics).lower()}",
            'source': random.choice(sources),
            'country': random.choice(countries),
            'topic': random.choice(topics),
            'timestamp': datetime.now() - timedelta(hours=random.randint(1, 168)),
            'sentiment_score': np.random.normal(0, 0.3),
            'engagement': random.randint(10, 10000),
            'credibility_score': np.random.beta(2, 1),
            'anti_narrative_probability': np.random.beta(1, 3),
            'influence_score': np.random.exponential(0.5)
        }
        headlines_data.append(headline_data)
    
    return pd.DataFrame(headlines_data)

@st.cache_data
def generate_time_series_data():
    """Generate time series data for trends"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    
    # Generate realistic trends with some seasonality
    sentiment_trend = []
    anti_narrative_trend = []
    donor_activity = []
    donation_amounts = []
    
    for i, date in enumerate(dates):
        # Add some seasonality and noise
        base_sentiment = 0.1 * np.sin(i * 2 * np.pi / 365) + np.random.normal(0, 0.2)
        base_anti_narrative = 0.3 + 0.1 * np.sin(i * 2 * np.pi / 52) + np.random.exponential(0.1)
        
        # Donor activity influenced by sentiment and current events
        donor_base = 1000 + 200 * np.sin(i * 2 * np.pi / 365) # Seasonal giving
        donor_sentiment_impact = donor_base * (1 + 0.3 * max(0, -base_sentiment)) # More giving when sentiment is negative
        donor_noise = np.random.normal(0, 50)
        daily_donors = max(0, donor_sentiment_impact + donor_noise)
        
        # Donation amounts also influenced by sentiment and anti-narrative
        base_amount = 150 + 50 * np.sin(i * 2 * np.pi / 365)
        crisis_multiplier = 1 + 0.5 * base_anti_narrative # More generous during crises
        amount_per_donor = base_amount * crisis_multiplier + np.random.normal(0, 20)
        
        sentiment_trend.append(base_sentiment)
        anti_narrative_trend.append(min(base_anti_narrative, 1.0))
        donor_activity.append(int(daily_donors))
        donation_amounts.append(max(10, amount_per_donor))
    
    return pd.DataFrame({
        'timestamp': dates,
        'sentiment': sentiment_trend,
        'anti_narrative_prob': anti_narrative_trend,
        'daily_donors': donor_activity,
        'avg_donation_amount': donation_amounts,
        'total_donations': np.array(donor_activity) * np.array(donation_amounts)
    })

@st.cache_data
def generate_donor_demographics():
    """Generate donor demographic and behavioral data"""
    np.random.seed(42)
    
    donor_segments = ['Young Professionals', 'Middle-aged Families', 'Retirees', 'High Net Worth', 'Corporate']
    causes = ['Education', 'Healthcare', 'Environment', 'Social Justice', 'Disaster Relief', 'Poverty Alleviation']
    
    demographics_data = []
    for segment in donor_segments:
        for cause in causes:
            # Generate segment-specific patterns
            if segment == 'Young Professionals':
                avg_donation = np.random.normal(75, 25)
                frequency = np.random.normal(8, 2)  # times per year
                sentiment_sensitivity = np.random.normal(0.8, 0.1)
            elif segment == 'Middle-aged Families':
                avg_donation = np.random.normal(150, 50)
                frequency = np.random.normal(6, 2)
                sentiment_sensitivity = np.random.normal(0.6, 0.1)
            elif segment == 'Retirees':
                avg_donation = np.random.normal(200, 60)
                frequency = np.random.normal(12, 3)
                sentiment_sensitivity = np.random.normal(0.4, 0.1)
            elif segment == 'High Net Worth':
                avg_donation = np.random.normal(2000, 800)
                frequency = np.random.normal(4, 1)
                sentiment_sensitivity = np.random.normal(0.3, 0.1)
            else:  # Corporate
                avg_donation = np.random.normal(5000, 2000)
                frequency = np.random.normal(2, 0.5)
                sentiment_sensitivity = np.random.normal(0.7, 0.1)
            
            demographics_data.append({
                'segment': segment,
                'cause': cause,
                'avg_donation': max(10, avg_donation),
                'frequency_per_year': max(1, frequency),
                'sentiment_sensitivity': max(0.1, min(1.0, sentiment_sensitivity)),
                'donor_count': np.random.randint(50, 500),
                'retention_rate': np.random.beta(3, 1),
                'growth_rate': np.random.normal(0.05, 0.02)
            })
    
    return pd.DataFrame(demographics_data)

def predict_donor_trends(historical_data, demographics_data, years_ahead=3):
    """Generate donor predictions based on historical data and trends"""
    
    # Prepare features for prediction
    historical_data['month'] = historical_data['timestamp'].dt.month
    historical_data['year'] = historical_data['timestamp'].dt.year
    historical_data['day_of_year'] = historical_data['timestamp'].dt.dayofyear
    
    # Features for prediction
    features = ['sentiment', 'anti_narrative_prob', 'month', 'day_of_year']
    
    # Predict daily donors
    X = historical_data[features].fillna(0)
    y_donors = historical_data['daily_donors']
    y_amounts = historical_data['avg_donation_amount']
    
    # Simple linear regression for demonstration
    model_donors = LinearRegression()
    model_amounts = LinearRegression()
    
    model_donors.fit(X, y_donors)
    model_amounts.fit(X, y_amounts)
    
    # Generate future predictions
    future_dates = pd.date_range(
        start=historical_data['timestamp'].max() + timedelta(days=1),
        periods=365 * years_ahead,
        freq='D'
    )
    
    future_predictions = []
    for i, date in enumerate(future_dates):
        # Simulate future sentiment and anti-narrative trends
        future_sentiment = 0.1 * np.sin(i * 2 * np.pi / 365) + np.random.normal(0, 0.1)
        future_anti_narrative = 0.3 + 0.1 * np.sin(i * 2 * np.pi / 52) + abs(np.random.normal(0, 0.05))
        
        future_features = [[
            future_sentiment,
            min(future_anti_narrative, 1.0),
            date.month,
            date.timetuple().tm_yday
        ]]
        
        predicted_donors = max(0, model_donors.predict(future_features)[0])
        predicted_amount = max(10, model_amounts.predict(future_features)[0])
        
        future_predictions.append({
            'date': date,
            'predicted_donors': int(predicted_donors),
            'predicted_avg_amount': predicted_amount,
            'predicted_total': predicted_donors * predicted_amount,
            'sentiment_forecast': future_sentiment,
            'anti_narrative_forecast': min(future_anti_narrative, 1.0)
        })
    
    return pd.DataFrame(future_predictions)

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">📊 Social Media Sentiment & Anti-Narrative Analyzer</h1>
        <p style="color: white; margin: 0; opacity: 0.9;">Real-time monitoring and analysis of social media narratives and sentiment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🔧 Analysis Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"]
    )
    
    # Platform filter
    platforms = st.sidebar.multiselect(
        "Platforms",
        ['Twitter/X', 'Facebook', 'Reddit', 'LinkedIn', 'News Sites', 'Blogs', 'Forums'],
        default=['Twitter/X', 'Facebook', 'Reddit']
    )
    
    # Topic filter
    topics = st.sidebar.multiselect(
        "Topics",
        ['Economic Policy', 'Climate Change', 'Technology', 'Healthcare', 'Education', 'Immigration'],
        default=['Economic Policy', 'Technology']
    )
    
    # Geographic filter
    countries = st.sidebar.multiselect(
        "Countries",
        ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'Brazil'],
        default=['USA', 'UK', 'Germany']
    )
    
    # Sensitivity threshold
    sensitivity = st.sidebar.slider(
        "Anti-Narrative Detection Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Load data
    df = generate_sample_data()
    time_series_df = generate_time_series_data()
    demographics_df = generate_donor_demographics()
    
    # Generate predictions
    predictions_df = predict_donor_trends(time_series_df, demographics_df)
    
    # Filter data based on selections
    if platforms:
        df = df[df['source'].isin(platforms)]
    if topics:
        df = df[df['topic'].isin(topics)]
    if countries:
        df = df[df['country'].isin(countries)]
    
    # Apply sensitivity threshold
    df['is_anti_narrative'] = df['anti_narrative_probability'] > sensitivity
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_headlines = len(df)
        st.metric(
            label="📰 Total Headlines Analyzed",
            value=f"{total_headlines:,}",
            delta=f"+{random.randint(10, 50)} in last hour"
        )
    
    with col2:
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_color = "🟢" if avg_sentiment > 0.1 else "🔴" if avg_sentiment < -0.1 else "🟡"
        st.metric(
            label=f"{sentiment_color} Average Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=f"{avg_sentiment - np.random.normal(0, 0.1):.3f} vs yesterday"
        )
    
    with col3:
        anti_narrative_count = df['is_anti_narrative'].sum()
        anti_narrative_pct = (anti_narrative_count / total_headlines) * 100 if total_headlines > 0 else 0
        alert_level = "🚨" if anti_narrative_pct > 30 else "⚠️" if anti_narrative_pct > 15 else "✅"
        st.metric(
            label=f"{alert_level} Anti-Narrative Detection",
            value=f"{anti_narrative_count} ({anti_narrative_pct:.1f}%)",
            delta=f"{random.randint(-5, 15)} from last period"
        )
    
    with col4:
        # Calculate donor metrics from time series
        recent_donors = time_series_df['daily_donors'].tail(30).mean()
        donor_growth = ((recent_donors - time_series_df['daily_donors'].head(30).mean()) / time_series_df['daily_donors'].head(30).mean()) * 100
        st.metric(
            label="📊 Avg Daily Donors",
            value=f"{recent_donors:.0f}",
            delta=f"{donor_growth:+.1f}% vs 30 days ago"
        )
    
    # Charts section
    st.header("📈 Real-time Analysis Dashboard")
    
    # Row 1: Sentiment Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        # Create sentiment categories
        df['sentiment_category'] = pd.cut(
            df['sentiment_score'],
            bins=[-np.inf, -0.2, 0.2, np.inf],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        sentiment_counts = df['sentiment_category'].value_counts()
        
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution Across All Sources",
            color_discrete_map={'Positive': '#00aa44', 'Neutral': '#ffaa00', 'Negative': '#ff4444'}
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Anti-Narrative Detection by Platform")
        platform_anti_narrative = df.groupby('source')['is_anti_narrative'].agg(['sum', 'count']).reset_index()
        platform_anti_narrative['percentage'] = (platform_anti_narrative['sum'] / platform_anti_narrative['count']) * 100
        
        fig_bar = px.bar(
            platform_anti_narrative,
            x='source',
            y='percentage',
            title="Anti-Narrative Content Percentage by Platform",
            color='percentage',
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Row 2: Geographic and Topic Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment by Country")
        country_sentiment = df.groupby('country')['sentiment_score'].mean().reset_index()
        
        fig_country = px.bar(
            country_sentiment,
            x='country',
            y='sentiment_score',
            title="Average Sentiment Score by Country",
            color='sentiment_score',
            color_continuous_scale='RdYlGn'
        )
        fig_country.update_layout(height=400, xaxis_tickangle=-45)
        fig_country.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        st.subheader("Topic Analysis")
        topic_analysis = df.groupby('topic').agg({
            'sentiment_score': 'mean',
            'anti_narrative_probability': 'mean',
            'engagement': 'sum'
        }).reset_index()
        
        fig_topic = px.scatter(
            topic_analysis,
            x='sentiment_score',
            y='anti_narrative_probability',
            size='engagement',
            color='topic',
            title="Topic Analysis: Sentiment vs Anti-Narrative Risk",
            hover_data=['engagement']
        )
        fig_topic.update_layout(height=400)
        fig_topic.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_topic.add_hline(y=sensitivity, line_dash="dash", line_color="red", opacity=0.7)
        st.plotly_chart(fig_topic, use_container_width=True)
    
    # Row 3: Donor Analytics and Predictions
    st.header("💰 Donor Analytics & Trend Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Donor Activity")
        
        # Resample to monthly for cleaner visualization
        monthly_donors = time_series_df.set_index('timestamp').resample('M').agg({
            'daily_donors': 'sum',
            'total_donations': 'sum',
            'sentiment': 'mean'
        }).reset_index()
        
        fig_donors = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Monthly Donor Count', 'Total Monthly Donations ($)'),
            vertical_spacing=0.1
        )
        
        fig_donors.add_trace(
            go.Scatter(x=monthly_donors['timestamp'], y=monthly_donors['daily_donors'],
                      mode='lines+markers', name='Donors', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig_donors.add_trace(
            go.Scatter(x=monthly_donors['timestamp'], y=monthly_donors['total_donations'],
                      mode='lines+markers', name='Donations ($)', line=dict(color='green')),
            row=2, col=1
        )
        
        fig_donors.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_donors, use_container_width=True)
    
    with col2:
        st.subheader("Donor Segment Analysis")
        
        # Aggregate demographics data
        segment_summary = demographics_df.groupby('segment').agg({
            'avg_donation': 'mean',
            'donor_count': 'sum',
            'retention_rate': 'mean',
            'growth_rate': 'mean'
        }).reset_index()
        
        segment_summary['total_value'] = segment_summary['avg_donation'] * segment_summary['donor_count']
        
        fig_segments = px.treemap(
            segment_summary,
            path=['segment'],
            values='total_value',
            color='retention_rate',
            title='Donor Segments by Total Value (Color = Retention Rate)',
            color_continuous_scale='Viridis'
        )
        fig_segments.update_layout(height=500)
        st.plotly_chart(fig_segments, use_container_width=True)
    
    # Prediction Section
    st.header("🔮 Donor Predictions (Next 3 Years)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Annual predictions summary
        annual_predictions = predictions_df.groupby(predictions_df['date'].dt.year).agg({
            'predicted_donors': 'sum',
            'predicted_total': 'sum'
        }).reset_index()
        annual_predictions['year'] = annual_predictions['date']
        
        current_year_total = time_series_df['total_donations'].sum()
        predicted_growth = ((annual_predictions['predicted_total'].iloc[0] - current_year_total) / current_year_total) * 100
        
        st.metric(
            label="📈 Predicted Annual Growth",
            value=f"{predicted_growth:.1f}%",
            delta=f"${annual_predictions['predicted_total'].iloc[0] - current_year_total:,.0f} increase"
        )
        
        st.metric(
            label="👥 Predicted Donor Growth",
            value=f"{((annual_predictions['predicted_donors'].iloc[0] - time_series_df['daily_donors'].sum()) / time_series_df['daily_donors'].sum()) * 100:.1f}%",
            delta=f"{annual_predictions['predicted_donors'].iloc[0] - time_series_df['daily_donors'].sum():.0f} new donors"
        )
    
    with col2:
        st.subheader("3-Year Projection")
        
        fig_projection = px.bar(
            annual_predictions,
            x='year',
            y='predicted_total',
            title='Predicted Annual Donation Totals',
            color='predicted_total',
            color_continuous_scale='Blues'
        )
        fig_projection.update_layout(height=300)
        st.plotly_chart(fig_projection, use_container_width=True)
    
    with col3:
        st.subheader("Seasonal Patterns")
        
        # Monthly pattern analysis
        monthly_pattern = predictions_df.groupby(predictions_df['date'].dt.month).agg({
            'predicted_donors': 'mean',
            'predicted_total': 'mean'
        }).reset_index()
        monthly_pattern['month_name'] = pd.to_datetime(monthly_pattern['date'], format='%m').dt.strftime('%b')
        
        fig_seasonal = px.line(
            monthly_pattern,
            x='month_name',
            y='predicted_total',
            title='Predicted Monthly Donation Pattern',
            markers=True
        )
        fig_seasonal.update_layout(height=300)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Detailed Predictions Table
    st.subheader("📊 Detailed Predictions Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Impact on Donations")
        
        # Correlation analysis
        correlation_data = pd.concat([
            time_series_df[['sentiment', 'anti_narrative_prob', 'total_donations']],
            predictions_df[['sentiment_forecast', 'anti_narrative_forecast', 'predicted_total']].rename(columns={
                'sentiment_forecast': 'sentiment',
                'anti_narrative_forecast': 'anti_narrative_prob',
                'predicted_total': 'total_donations'
            })
        ])
        
        fig_correlation = px.scatter(
            correlation_data,
            x='sentiment',
            y='total_donations',
            size='anti_narrative_prob',
            title='Sentiment vs Donations (Size = Anti-Narrative Risk)',
            opacity=0.6
        )
        fig_correlation.update_layout(height=400)
        st.plotly_chart(fig_correlation, use_container_width=True)
    
    with col2:
        st.subheader("Cause-Specific Predictions")
        
        # Predict by cause category
        cause_predictions = demographics_df.groupby('cause').agg({
            'avg_donation': 'mean',
            'donor_count': 'sum',
            'growth_rate': 'mean'
        }).reset_index()
        
        cause_predictions['predicted_2025_total'] = (
            cause_predictions['avg_donation'] * 
            cause_predictions['donor_count'] * 
            (1 + cause_predictions['growth_rate']) * 365
        )
        
        fig_causes = px.bar(
            cause_predictions,
            x='cause',
            y='predicted_2025_total',
            title='2025 Predictions by Cause',
            color='growth_rate',
            color_continuous_scale='RdYlGn'
        )
        fig_causes.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_causes, use_container_width=True)
    
    # Risk Assessment
    st.header("⚠️ Risk Assessment & Scenario Planning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card alert-low">
            <h4>🟢 Optimistic Scenario (+15%)</h4>
            <ul>
                <li><strong>Conditions:</strong> Positive sentiment, low anti-narrative</li>
                <li><strong>Donation Growth:</strong> +15% annually</li>
                <li><strong>Donor Retention:</strong> 85%+</li>
                <li><strong>Key Drivers:</strong> Economic stability, trust in institutions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card alert-medium">
            <h4>🟡 Baseline Scenario (+5%)</h4>
            <ul>
                <li><strong>Conditions:</strong> Neutral sentiment, moderate anti-narrative</li>
                <li><strong>Donation Growth:</strong> +5% annually</li>
                <li><strong>Donor Retention:</strong> 70-75%</li>
                <li><strong>Key Drivers:</strong> Current trends continue</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card alert-high">
            <h4>🔴 Pessimistic Scenario (-10%)</h4>
            <ul>
                <li><strong>Conditions:</strong> Negative sentiment, high anti-narrative</li>
                <li><strong>Donation Growth:</strong> -10% annually</li>
                <li><strong>Donor Retention:</strong> <60%</li>
                <li><strong>Key Drivers:</strong> Economic uncertainty, mistrust</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Analytics
    st.header("🔬 Advanced Donor Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Donor Lifetime Value Prediction")
        
        # Calculate CLV for each segment
        demographics_df['predicted_clv'] = (
            demographics_df['avg_donation'] * 
            demographics_df['frequency_per_year'] * 
            (demographics_df['retention_rate'] / (1 - demographics_df['retention_rate']))
        )
        
        clv_summary = demographics_df.groupby('segment')['predicted_clv'].mean().reset_index()
        
        fig_clv = px.bar(
            clv_summary,
            x='segment',
            y='predicted_clv',
            title='Predicted Customer Lifetime Value by Segment',
            color='predicted_clv',
            color_continuous_scale='Plasma'
        )
        fig_clv.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_clv, use_container_width=True)
    
    with col2:
        st.subheader("Churn Risk Analysis")
        
        # Simulate churn risk based on sentiment and engagement
        churn_risk = demographics_df.copy()
        churn_risk['churn_probability'] = (
            (1 - churn_risk['retention_rate']) * 
            (1 + churn_risk['sentiment_sensitivity'] * abs(time_series_df['sentiment'].mean()))
        )
        
        churn_summary = churn_risk.groupby('segment')['churn_probability'].mean().reset_index()
        
        fig_churn = px.bar(
            churn_summary,
            x='segment',
            y='churn_probability',
            title='Predicted Churn Risk by Segment',
            color='churn_probability',
            color_continuous_scale='Reds'
        )
        fig_churn.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_churn, use_container_width=True)
    
    # Detailed Analysis Section
    st.header("🔍 Detailed Analysis")
    
    # Top influential content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Most Influential Content")
        top_influential = df.nlargest(10, 'influence_score')[['headline', 'source', 'influence_score', 'engagement']]
        
        for idx, row in top_influential.iterrows():
            influence_bar = "🔴" if row['influence_score'] > 1.5 else "🟡" if row['influence_score'] > 1.0 else "🟢"
            st.write(f"{influence_bar} **{row['source']}** | Score: {row['influence_score']:.2f} | Engagement: {row['engagement']:,}")
            st.write(f"_{row['headline'][:100]}..._")
            st.write("---")
    
    with col2:
        st.subheader("⚠️ High-Risk Anti-Narrative Content")
        high_risk = df[df['anti_narrative_probability'] > 0.7].nlargest(10, 'anti_narrative_probability')
        
        if len(high_risk) > 0:
            for idx, row in high_risk.iterrows():
                risk_level = "🚨" if row['anti_narrative_probability'] > 0.9 else "⚠️"
                st.write(f"{risk_level} **{row['source']}** | Risk: {row['anti_narrative_probability']:.3f} | {row['country']}")
                st.write(f"_{row['headline'][:100]}..._")
                st.write("---")
        else:
            st.info("No high-risk anti-narrative content detected with current sensitivity settings.")
    
    # Network Analysis Simulation
    st.header("🕸️ Network Analysis")
    
    # Simulate network data
    network_data = df.groupby(['source', 'country']).agg({
        'engagement': 'sum',
        'influence_score': 'mean',
        'anti_narrative_probability': 'mean'
    }).reset_index()
    
    fig_network = px.scatter(
        network_data,
        x='engagement',
        y='influence_score',
        size='anti_narrative_probability',
        color='source',
        hover_data=['country'],
        title="Content Network: Engagement vs Influence (Size = Anti-Narrative Risk)",
        log_x=True
    )
    fig_network.update_layout(height=500)
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Alerts and Recommendations - Enhanced
    st.header("🚨 Enhanced Alerts & Strategic Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card alert-high">
            <h4>🚨 Critical Donor Alerts</h4>
            <ul>
                <li>High-value donor churn risk detected</li>
                <li>Negative sentiment spike in key demographics</li>
                <li>Corporate donors showing reduced engagement</li>
                <li>Anti-narrative content affecting donor confidence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card alert-medium">
            <h4>⚠️ Strategic Opportunities</h4>
            <ul>
                <li>Young professionals segment showing growth potential</li>
                <li>Environmental causes gaining traction</li>
                <li>Seasonal giving patterns favor Q4 campaigns</li>
                <li>Social media sentiment improving for health causes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card alert-low">
            <h4>💡 AI-Powered Recommendations</h4>
            <ul>
                <li>Increase engagement during high-sentiment periods</li>
                <li>Target retiree segment for stable growth</li>
                <li>Deploy counter-narrative strategies proactively</li>
                <li>Optimize donation timing based on predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Export Section
    st.header("📊 Data Export & API Access")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Current Analysis")
        if st.button("📥 Download Full Report (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
    
    with col2:
        st.subheader("API Integration Status")
        st.info("🟢 Twitter API: Connected")
        st.info("🟢 News API: Connected") 
        st.info("🟡 Reddit API: Rate Limited")
        st.info("🔴 Facebook API: Authentication Required")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p><strong>Advanced Social Media Sentiment & Donor Analytics Platform</strong> | 
        Last Updated: {}</p>
        <p>This comprehensive demo showcases real-time sentiment analysis, donor behavior prediction, 
        and strategic fundraising insights. Real implementation includes live API integrations, 
        machine learning models, and production-grade infrastructure.</p>
        <p><em>🔮 Powered by AI-driven predictive analytics and real-time data processing</em></p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
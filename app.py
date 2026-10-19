import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StockLens | Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 2. WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #10283a);
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #091522;
}

h1, h2, h3, p, label {
    color: #eef5ff;
}

div.stButton > button {
    background-color: #35c6a4;
    color: #07111f;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    padding: 8px 20px;
}

div.stButton > button:hover {
    background-color: #6ce5c7;
    color: #07111f;
}

[data-testid="stMetric"] {
    background-color: #10283a;
    border: 1px solid #25465b;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOGIN SYSTEM
# ==========================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.title("📈 StockLens")
    st.subheader("Stock Market Analysis")
    st.write(
        "Explore stock prices, market trends, "
        "interactive charts and machine learning results."
    )

    st.divider()

    st.subheader("🔐 Welcome Back")

    with st.form("stocklens_login", clear_on_submit=False):

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        login_button = st.form_submit_button(
            "Login",
            use_container_width=True
        )

    if login_button:

        clean_username = username.strip().lower()

        if clean_username == "shifa" and password == "12345":
            st.session_state.authenticated = True
            st.rerun()

        else:
            st.error(
                "Incorrect username or password. "
                "Please enter the demo credentials shown below."
            )

    st.info(
        "Demo username: shifa\n\n"
        "Demo password: 12345"
    )

    st.caption(
        "Student project demonstration. "
        "This login is not secure production authentication."
    )

    st.stop()

# ==========================================
# 4. CREATE SAMPLE STOCK DATA
# ==========================================

@st.cache_data
def create_stock_data():

    rng = np.random.default_rng(42)

    number_of_records = 260

    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=number_of_records,
        freq="B"
    )

    close = 180 + np.cumsum(
        rng.normal(0.15, 1.8, number_of_records)
    )

    open_price = close + rng.normal(
        0, 1, number_of_records
    )

    high = np.maximum(open_price, close) + rng.uniform(
        0.2, 2, number_of_records
    )

    low = np.minimum(open_price, close) - rng.uniform(
        0.2, 2, number_of_records
    )

    volume = rng.integers(
        500000, 4500000, number_of_records
    )

    data = pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })

    data["Daily_Return"] = data["Close"].pct_change() * 100

    data["Price_Change"] = (
        data["Close"] - data["Open"]
    )

    data["High_Low_Range"] = (
        data["High"] - data["Low"]
    )

    data["Trend"] = (
        data["Close"] > data["Open"]
    ).astype(int)

    return data


df = create_stock_data()

# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("📈 StockLens")

st.sidebar.write("Stock Market Analysis")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Stock Data",
        "ML Results",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Python | Data Science | Machine Learning"
)

if st.sidebar.button("Logout", use_container_width=True):

    st.session_state.authenticated = False
    st.rerun()

# ==========================================
# 6. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("📊 Market Dashboard")

    st.write(
        "Welcome to your stock market analytics workspace."
    )

    st.warning(
        "This dashboard uses generated sample data, "
        "not live stock market prices."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Close",
        f"{df['Close'].iloc[-1]:.2f}"
    )

    col2.metric(
        "Average Close",
        f"{df['Close'].mean():.2f}"
    )

    col3.metric(
        "Highest Close",
        f"{df['Close'].max():.2f}"
    )

    col4.metric(
        "Trading Records",
        f"{len(df)}"
    )

    st.divider()

    st.subheader("📈 Closing Price Trend")

    fig = px.line(
        df,
        x="Date",
        y="Close",
        title="Stock Closing Price Over Time",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.subheader("📊 Trading Volume")

        fig = px.area(
            df,
            x="Date",
            y="Volume",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("📉 Daily Return Distribution")

        fig = px.histogram(
            df.dropna(subset=["Daily_Return"]),
            x="Daily_Return",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("📋 Recent Stock Records")

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )

# ==========================================
# 7. STOCK DATA PAGE
# ==========================================

elif page == "Stock Data":

    st.title("📁 Stock Data Explorer")

    st.write(
        "Explore the sample dataset and its statistical summary."
    )

    st.subheader("Stock Market Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("📊 Descriptive Statistics")

    st.dataframe(
        df.describe(numeric_only=True),
        use_container_width=True
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Stock Data as CSV",
        data=csv_data,
        file_name="stock_market_data.csv",
        mime="text/csv"
    )

# ==========================================
# 8. MACHINE LEARNING RESULTS
# ==========================================

elif page == "ML Results":

    st.title("🤖 Machine Learning Results")

    st.write(
        "Compare Logistic Regression and Decision Tree "
        "classification models."
    )

    st.info(
        "The models classify same-day upward or downward "
        "movement using sample data. They do not predict "
        "future stock prices."
    )

    # Avoid using Price_Change as a feature because it
    # directly reveals whether Close is greater than Open.

    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "High_Low_Range"
    ]

    X = df[features]
    y = df["Trend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        ) * 100

        results.append({
            "Model": model_name,
            "Accuracy (%)": round(accuracy, 2)
        })

        trained_models[model_name] = model

    results_df = pd.DataFrame(results)

    st.subheader("Model Accuracy Comparison")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    fig = px.bar(
        results_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)",
        title="Classification Model Accuracy",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    selected_model_name = st.selectbox(
        "Select a model to inspect",
        list(trained_models.keys())
    )

    selected_model = trained_models[selected_model_name]

    predictions = selected_model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    st.subheader(
        f"Confusion Matrix: {selected_model_name}"
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Down/Not Up", "Actual Up"],
        columns=["Predicted Down/Not Up", "Predicted Up"]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.caption(
        "These are demonstration results from generated data. "
        "They may differ from results in your internship report."
    )

# ==========================================
# 9. ABOUT PAGE
# ==========================================

elif page == "About":

    st.title("ℹ️ About StockLens")

    st.subheader("Project Information")

    st.write(
        "**Project Title:** Stock Market Data Analysis "
        "and Trend Classification Using Python "
        "and Machine Learning"
    )

    st.write(
        "**Domain:** Data Science and Machine Learning"
    )

    st.write(
        "**Programming Language:** Python"
    )

    st.write(
        "**Libraries:** Pandas, NumPy, Plotly and Scikit-learn"
    )

    st.write(
        "**Framework:** Streamlit"
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Explore stock market data.
    - Calculate daily returns and price ranges.
    - Visualize closing prices and trading volume.
    - Compare classification models.
    - Display model accuracy and confusion matrices.
    """)

    st.warning(
        "This demonstration currently generates sample data. "
        "It is not connected to a live stock market API."
    )

    st.success(
        "StockLens demonstrates stock data analysis, "
        "visualization and machine learning."
    )

# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "StockLens | Student Project | Python and Machine Learning"
)

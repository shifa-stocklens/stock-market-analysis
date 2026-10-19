import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="StockLens",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #10283a);
    color: white;
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
}
</style>
""", unsafe_allow_html=True)

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    left, right = st.columns(2)

    with left:
        st.title("📈 StockLens")
        st.header("Stock Market Analysis")
        st.write(
            "Explore stock prices, market trends, "
            "interactive charts and machine learning results."
        )

    with right:
        st.subheader("Welcome Back")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input(
                "Password", type="password"
            )
            submit = st.form_submit_button("Login")

        st.caption("Demo username: shifa")
        st.caption("Demo password: 12345")

        if submit:
            if username == "shifa" and password == "12345":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    st.stop()

st.sidebar.title("📈 StockLens")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Stock Data", "ML Results", "About"]
)

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

@st.cache_data
def create_data():
    rng = np.random.default_rng(42)
    n = 260

    dates = pd.date_range(
        end=pd.Timestamp.today(),
        periods=n,
        freq="B"
    )

    close = 180 + np.cumsum(
        rng.normal(0.15, 1.8, n)
    )

    open_price = close + rng.normal(0, 1, n)

    high = np.maximum(open_price, close) + rng.uniform(
        0.2, 2, n
    )

    low = np.minimum(open_price, close) - rng.uniform(
        0.2, 2, n
    )

    volume = rng.integers(500000, 4500000, n)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })

df = create_data()

df["Daily_Return"] = df["Close"].pct_change() * 100
df["Price_Change"] = df["Close"] - df["Open"]
df["High_Low_Range"] = df["High"] - df["Low"]
df["Trend"] = (df["Close"] > df["Open"]).astype(int)

if page == "Dashboard":

    st.title("📊 Market Dashboard")
    st.write("Your stock market analytics workspace")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Latest Close", f"{df['Close'].iloc[-1]:.2f}")
    c2.metric("Average Close", f"{df['Close'].mean():.2f}")
    c3.metric("Highest Close", f"{df['Close'].max():.2f}")
    c4.metric("Trading Records", len(df))

    st.subheader("Closing Price Trend")

    fig = px.line(
        df, x="Date", y="Close",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.area(
            df, x="Date", y="Volume",
            title="Trading Volume",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df, x="Daily_Return",
            title="Daily Return Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Stock Data":

    st.title("📁 Stock Data")

    st.dataframe(df, use_container_width=True)

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(numeric_only=True),
        use_container_width=True
    )

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "stock_data.csv",
        "text/csv"
    )

elif page == "ML Results":

    st.title("🤖 Machine Learning Results")

    features = [
        "Open", "High", "Low", "Volume",
        "Price_Change", "High_Low_Range"
    ]

    X = df[features]
    y = df["Trend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=42
        )
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test, predictions
        ) * 100

        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy, 2)
        })

    result_df = pd.DataFrame(results)

    st.dataframe(result_df, use_container_width=True)

    fig = px.bar(
        result_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        title="Model Accuracy Comparison",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "These models classify same-day upward or downward "
        "movement using demonstration data. They do not "
        "predict future stock prices."
    )

elif page == "About":

    st.title("About This Project")

    st.write(
        "**Project:** Stock Market Data Analysis "
        "and Trend Classification Using Python "
        "and Machine Learning"
    )

    st.write("**Domain:** Data Science and Machine Learning")

    st.write(
        "**Technologies:** Python, Pandas, NumPy, "
        "Plotly, Scikit-learn and Streamlit"
    )

    st.write(
        "The application demonstrates data analysis, "
        "visualization, classification and model evaluation."
    )

    st.warning(
        "This version uses generated sample data, "
        "not live stock market prices."
    )

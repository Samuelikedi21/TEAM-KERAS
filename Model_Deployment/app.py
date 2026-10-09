
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from html import escape


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Market Basket Grouping",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent



# LOAD TRAINED MODEL

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()


CLUSTER_INFO = {
    0: (
        "Single-item buyer",
        "Typically one product in one category, "
        "averaging about $200 per transaction."
    ),
    1: (
        "Multi-category buyer",
        "Typically about 3 products across 2 categories, "
        "averaging about $442 per transaction."
    ),
}



# CUSTOM CSS

st.html("""
<style>
    /* Overall application */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    /* Main content */
    .block-container {
        max-width: 1250px;
        padding-top: 4.5rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111b2e;
        border-right: 1px solid #26354d;
        min-width: 320px;
        max-width: 320px;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.3rem;
        padding-right: 1.3rem;
    }

    /* Sidebar branding */
    .brand {
        color: #60a5fa;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.4;
        margin-bottom: 8px;
        overflow-wrap: break-word;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }

    /* Page section label */
    .section-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    /* Main hero banner */
    .hero {
        background: linear-gradient(
            115deg,
            #172554,
            #1e3a8a,
            #172554
        );
        border: 1px solid #334e94;
        border-radius: 20px;
        padding: 36px;
        margin: 15px 0 30px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        line-height: 1.2;
        margin: 0 0 15px;
    }

    .hero p {
        font-size: 16px;
        color: #cbd5e1;
        line-height: 1.8;
        margin: 0;
    }

    /* Dashboard statistics */
    .stat-card {
        background: #152137;
        border: 1px solid #293b54;
        border-radius: 15px;
        padding: 24px;
        min-height: 125px;
        margin-bottom: 15px;
    }

    .stat-label {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 12px;
    }

    .stat-value {
        font-size: 27px;
        font-weight: 800;
        color: #f8fafc;
    }

    /* Quick guide */
    .info-card {
        background: #152137;
        border: 1px solid #293b54;
        border-radius: 15px;
        padding: 24px;
        margin-top: 12px;
        color: #cbd5e1;
        line-height: 1.8;
    }

    .info-card strong {
        color: #f8fafc;
    }

    .info-card p {
        margin: 5px 0 20px;
        color: #94a3b8;
    }

    /* BLUE PREDICTION RESULT CARD */
    .result-card {
        background: linear-gradient(
            115deg,
            #172554,
            #1e3a8a,
            #172554
        );
        border: 1px solid #3b82f6;
        border-radius: 18px;
        padding: 32px;
        margin-top: 20px;
        box-shadow: 0 8px 30px rgba(
            37, 99, 235, 0.10
        );
    }

    /* Blue analysis complete label */
    .result-label {
        color: #93c5fd;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    /* Predicted cluster name */
    .result-title {
        font-size: 29px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 14px;
        margin-bottom: 12px;
    }

    /* Predicted cluster description */
    .result-description {
        color: #dbeafe;
        font-size: 16px;
        line-height: 1.8;
    }

    /* Analyze button */
    div.stFormSubmitButton > button {
        background: linear-gradient(
            90deg,
            #2563eb,
            #3b82f6
        );
        color: #ffffff;
        border: none;
        border-radius: 10px;
        min-height: 50px;
        font-size: 16px;
        font-weight: 700;
        width: 100%;
    }

    div.stFormSubmitButton > button:hover {
        background: #1d4ed8;
        color: #ffffff;
        border: none;
    }

    /* Number input fields */
    [data-testid="stNumberInput"] input {
        border-radius: 9px;
    }

    hr {
        border-color: #293b54;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 24px;
        }

        .hero h1 {
            font-size: 28px;
        }

        .result-title {
            font-size: 24px;
        }
    }
</style>
""")



# SIDEBAR

with st.sidebar:

    st.html(
        '<div class="brand">Market Basket Grouping</div>'
    )

    st.caption("TRANSACTION INTELLIGENCE")

    st.divider()

    st.markdown("### Dashboard")
    st.markdown("**Transaction Predictor**")
    st.caption("AI-powered customer segmentation")

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** KMeans Clustering")
    st.write("**Segments:** 2")
    st.write("**Input Features:** 4")

    st.divider()

    st.markdown("### Available Segments")

    st.write("Single-item buyer")
    st.write("Multi-category buyer")

    st.divider()

    st.caption(
        "Market Basket Grouping | ML Project"
    )



# MAIN PAGE HEADER

st.html("""
<div class="section-label">
Analytics / Predictions
</div>
""")

st.html("""
<div class="hero">
    <h1>Transaction Predictor</h1>
    <p>
        Understand customer purchasing behavior
        using machine learning. Enter transaction
        details to discover which customer segment
        best matches the transaction.
    </p>
</div>
""")



# DASHBOARD OVERVIEW

col1, col2, col3 = st.columns(3)

with col1:
    st.html("""
    <div class="stat-card">
        <div class="stat-label">
            Machine Learning Model
        </div>
        <div class="stat-value">KMeans</div>
    </div>
    """)

with col2:
    st.html("""
    <div class="stat-card">
        <div class="stat-label">
            Customer Segments
        </div>
        <div class="stat-value">02</div>
    </div>
    """)

with col3:
    st.html("""
    <div class="stat-card">
        <div class="stat-label">
            Prediction Features
        </div>
        <div class="stat-value">04</div>
    </div>
    """)



# TRANSACTION ANALYSIS

st.markdown("## Transaction Analysis")

st.caption(
    "Provide the transaction information below "
    "to generate a customer segment prediction."
)

left, right = st.columns([2, 1], gap="large")



# TRANSACTION INPUT FORM

with left:

    with st.container(border=True):

        st.markdown("### Transaction Details")

        st.caption(
            "Complete the four fields below."
        )

        with st.form("transaction_form"):

            input_col1, input_col2 = st.columns(2)

            with input_col1:

                total_products = st.number_input(
                    "Total products",
                    min_value=1,
                    value=3,
                    step=1
                )

                total_units = st.number_input(
                    "Total units",
                    min_value=1,
                    value=2,
                    step=1
                )

            with input_col2:

                total_revenue = st.number_input(
                    "Total revenue ($)",
                    min_value=0.0,
                    value=61.0,
                    step=1.0
                )

                categories = st.number_input(
                    "Number of categories",
                    min_value=1,
                    value=2,
                    step=1
                )

            st.write("")

            submitted = st.form_submit_button(
                "Analyze Transaction",
                use_container_width=True
            )



# QUICK GUIDE

with right:

    st.markdown("### Quick Guide")

    st.html("""
    <div class="info-card">

        <strong>1. Enter your data</strong>
        <p>Fill in all four transaction fields.</p>

        <strong>2. Run the analysis</strong>
        <p>Click Analyze Transaction.</p>

        <strong>3. View your segment</strong>
        <p>
            See the predicted customer group
            and its typical behavior.
        </p>

    </div>
    """)



# MODEL PREDICTION
if submitted:

    new_data = pd.DataFrame([{
        "total_products": total_products,
        "total_units": total_units,
        "total_revenue": total_revenue,
        "categories": categories,
    }])

    try:

        # Use original saved scaler
        scaled = scaler.transform(new_data)

        # Predict with original KMeans model
        cluster = int(model.predict(scaled)[0])

        # Match prediction to customer segment
        name, description = CLUSTER_INFO.get(
            cluster,
            (
                f"Segment {cluster}",
                "A transaction segment identified "
                "by the KMeans model."
            )
        )

        st.markdown("## Prediction Results")

        safe_title = escape(
            f"Cluster {cluster}: {name}"
        )

        safe_description = escape(description)

        # Blue prediction card
        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                ANALYSIS COMPLETE
            </div>

            <div class="result-title">
                {safe_title}
            </div>

            <div class="result-description">
                {safe_description}
            </div>

        </div>
        """)

        st.write("")

        # Transaction summary
        st.markdown("### Transaction Summary")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Products",
            total_products
        )

        metric2.metric(
            "Units",
            total_units
        )

        metric3.metric(
            "Revenue",
            f"${total_revenue:,.2f}"
        )

        metric4.metric(
            "Categories",
            categories
        )

        st.caption(
            "The customer segment is assigned using "
            "the trained KMeans model and saved "
            "feature scaler."
        )

    except Exception as error:

        st.error(
            "The prediction could not be completed."
        )

        st.exception(error)


# FOOTER
st.divider()

st.caption(
    "Market Basket Grouping | Transaction Predictor "
    "• Powered by Python, Streamlit & Scikit-learn"
)

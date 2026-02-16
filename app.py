import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import plotly.graph_objects as go
import sklearn
import shap
st.set_page_config(page_title="Data Quality Scoring System",layout='wide')
st.title("Data Quality Scoring System")
st.markdown(
    """
    <style>
    
    /* Make full app background WHITE */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    /* Optional: reduce top spacing */
    .block-container {
        padding-top: 2rem;
    }
    /* Soft white dashboard cards */
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .section-title {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_models():
    
    quality_model = joblib.load("notebooks/quality_model.pkl")
    return quality_model
def outlier_detection(df):
    n1=0
    for col in num_col:
        q1 = df[col].quantile(0.25)
        q2=df[col].quantile(0.75)
        iqr = q2-q1
        lower = q1-(1.5*iqr)
        upper = q2+(1.5*iqr)
        n = ((df[col]<lower)|(df[col]>upper)).sum()
        n1+=n
    return n1
def detect_skewness(df):
    
    skewed_df= df[num_col].skew().abs().to_frame(name="skewness").sort_values("skewness",ascending=False)
    highly_skewed_count = skewed_df[skewed_df['skewness']>2]['skewness'].count()
    highly_skewed_col = skewed_df.iloc[0].name

    return highly_skewed_count,highly_skewed_col

# -------- Gauge Meter Function --------
def create_gauge(score):
    """
    Creates a circular gauge meter using Plotly.
    score: integer between 0 and 100
    """
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 50, 'color': "black"}},  # Score text styling
        title={'text': "Data Quality Score", 'font': {'size': 20, 'color': "black"}},
        
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            
            # Colored ranges (quality zones)
            'steps': [
                {'range': [0, 40], 'color': "#ff4b4b"},   # Red
                {'range': [40, 70], 'color': "#ffa500"},  # Orange
                {'range': [70, 100], 'color': "#00cc96"}  # Green
            ],
            
            # Needle / pointer color
            'bar': {'color': "white", 'thickness': 0.25},
            
            # Optional threshold marker
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    # Transparent background to match dark theme
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # transparent background
        font={'color': "black"},
    
        # 👇 These lines fix the huge stretched gauge
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,  # controls vertical size
         # subtle shadow look
        shapes=[
            dict(
                type='circle',
                xref='paper', yref='paper',
                x0=0.15, y0=0.15, x1=0.85, y1=0.85,
                fillcolor='rgba(255,255,255,0.03)',
                line_color='rgba(0,0,0,0)'
            )
        ]
    )

    return fig

def quality_classify(score):
    if score>=80:
        return "Good"
    elif score>=50:
        return "Moderate"
    else:
        return "Poor"
c1,c2=st.columns([1,2])
uploaded_file = c1.file_uploader("Upload csv file",type=["csv"],width=400)
if uploaded_file is not None:
    quality_model = load_models()
    df = pd.read_csv(uploaded_file)
    c2.dataframe(df.head(),width="content",use_container_width=True)
    
    num_cols = df.select_dtypes(include='number')
    num_col = [c for c in num_cols]
    cat_col = [c for c in df.columns if c not in num_col]
    
    missing_pct = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])
    duplicates_pct = df.duplicated().sum()/len(df)
    
    if len(num_col)>0:
        outliers_sum = outlier_detection(df)
        skewed_count,skewed_col = detect_skewness(df)
        outlier_pct = outliers_sum/(len(df)*len(num_col))
        noise_level = df[num_col].std().mean() / (df[num_col].mean().mean() + 1e-5)
        noise_level = min(noise_level, 1)
    else:
        outlier_pct = 0
        noise_level = 0
        skewed_count=0
    st.markdown(" #### Quality Outbreak")
    c1,c2,c3=st.columns([1,1,2])
    
    with c1:
        st.markdown(f"Rows:{df.shape[0]}|Columns:{df.shape[1]}")
        st.markdown(f"No. numerical column: {len(num_col)}")
        st.markdown(f"No. categorical column: {len(cat_col)}")
        st.markdown(f"Missing Data : {round(missing_pct*100,2)}%")
        
    with c2:
        
        st.write(f"Duplicate Rows : {round(duplicates_pct*100,2)} %")
        st.write(f"Outliers Detected : {round(outlier_pct *100,2)} %")
        st.write(f"No. of features highly skewed: {skewed_count}")
        if skewed_count>0:
            st.write(f"Highly skewed Feature: '{skewed_col}' column")
      
    drift_score =0 # reference data is not taken
    leakage_score=0 # target column is not specified

    input_value_quality = pd.DataFrame({
        "missing_%": [missing_pct*100],
        "outlier_%": [outlier_pct*100],
        "duplicates_%": [duplicates_pct*100],
        "noise_level": [noise_level],
        "drift_score": [drift_score],
        'leakage_score':[leakage_score]
    })
    input_value_quality["Unnamed: 0"] = 0
    input_value_quality = input_value_quality[
        quality_model.feature_names_in_
    ]

    score = int(quality_model.predict(input_value_quality))
    with c3:
        gauge_fig = create_gauge(score)
        st.plotly_chart(gauge_fig, use_container_width=False)
        label = quality_classify(score)
        st.subheader(label,text_alignment="center")
        with st.expander("Feature impact on Quality Score"):
            explainer = shap.Explainer(quality_model)
            shap_values = explainer(input_value_quality)
            fig,ax = plt.subplots()
            shap.plots.waterfall(shap_values[0],show=False)
            st.pyplot(fig)







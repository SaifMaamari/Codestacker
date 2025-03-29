import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
import joblib
import fitz  # PyMuPDF

# Function to assign severity level based on crime category
def get_severity(cat):
    severity_map = {
        1: ["NON-CRIMINAL", "SUSPICIOUS OCCURRENCE", "MISSING PERSON", "RUNAWAY", "RECOVERED VEHICLE"],
        2: ["WARRANTS", "OTHER OFFENSES", "VANDALISM", "TRESPASS", "DISORDERLY CONDUCT", "BAD CHECKS"],
        3: ["LARCENY/THEFT", "VEHICLE THEFT", "FORGERY/COUNTERFEITING", "DRUG/NARCOTIC", "STOLEN PROPERTY", "FRAUD", "BRIBERY", "EMBEZZLEMENT"],
        4: ["ROBBERY", "WEAPON LAWS", "BURGLARY", "EXTORTION"],
        5: ["KIDNAPPING", "ARSON"]
    }
    for sev, cats in severity_map.items():
        if cat.upper() in cats:
            return sev
    return "Unknown"

# Load dataset
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "Competition_Dataset.csv")
    df = pd.read_csv(file_path)
    df['Dates'] = pd.to_datetime(df['Dates'])
    return df

df = load_data()

# Sidebar: Filters
st.sidebar.title("CityX Crime Filters")
min_date = df['Dates'].min().date()
max_date = df['Dates'].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
crime_categories = df['Category'].unique().tolist()
selected_categories = st.sidebar.multiselect("Select Crime Category", crime_categories, default=crime_categories)
districts = df['PdDistrict'].dropna().unique().tolist()
selected_district = st.sidebar.selectbox("Select Police District", ["All"] + districts)

# Apply Filters
filtered_df = df[
    (df['Dates'].dt.date >= date_range[0]) & (df['Dates'].dt.date <= date_range[1]) &
    (df['Category'].isin(selected_categories))
]
if selected_district != "All":
    filtered_df = filtered_df[filtered_df['PdDistrict'] == selected_district]

# Show filtered dataset
st.write("### Filtered Crime Data")
st.dataframe(filtered_df.head())

# CORRECT COORDINATES FOR SAN FRANCISCO
lat_col, lon_col = "Latitude (Y)", "Longitude (X)"

# Heatmap Toggle
st.write("### Crime Hotspots in CityX (Map)")
show_heatmap = st.checkbox("Show Heatmap", value=True)

if lat_col in df.columns and lon_col in df.columns:
    df_map = filtered_df.dropna(subset=[lat_col, lon_col])
    map_center = [df_map[lon_col].mean(), df_map[lat_col].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12)

    if show_heatmap:
        HeatMap(df_map[[lon_col, lat_col]].values, radius=12).add_to(crime_map)
    else:
        for _, row in df_map.iterrows():
            folium.Marker(
                location=[row[lon_col], row[lat_col]],  # ✅ Correct Lat, Lon
                popup=row['Category'],
                icon=folium.Icon(color="red")
            ).add_to(crime_map)

    folium_static(crime_map)
else:
    st.write("\U0001F6A8 Error: Could not find Latitude/Longitude columns.")

# Crime Trends Chart
st.write("### Crime Trends Over Time")
if 'df_map' in locals():
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_map['Dates'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Crime Count")
    st.pyplot(fig)
else:
    st.warning("Map data not available for trend visualization.")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("crime_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Predict from user input
st.markdown("---")
st.header("\U0001F50D Predict Crime Category")
user_input = st.text_area("Enter a description of the incident:", "")
if user_input:
    input_transformed = vectorizer.transform([user_input])
    prediction = model.predict(input_transformed)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"**Predicted Crime Category:** {predicted_label}")
    severity = get_severity(predicted_label)
    st.info(f"**Predicted Severity Level:** {severity}")

# PDF Upload and Prediction
st.markdown("---")
st.header("\U0001F4C4 Extract & Predict from Police Report (PDF)")
uploaded_pdf = st.file_uploader("Upload a PDF crime report", type="pdf")

if uploaded_pdf is not None:
    st.write("\u2705 PDF Uploaded Successfully!")
    try:
        with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()

        if text.strip():
            st.write("\u2705 PDF Text Extracted Successfully!")
            st.write("### Extracted Text:")
            st.write(text[:1000] + "...")
            input_transformed = vectorizer.transform([text])
            prediction = model.predict(input_transformed)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"**Predicted Crime Category:** {predicted_label}")
            severity = get_severity(predicted_label)
            st.info(f"**Predicted Severity Level:** {severity}")
        else:
            st.error("\U0001F6A8 PDF uploaded, but no text was extracted!")
    except Exception as e:
        st.error(f"\U0001F6A8 Error processing PDF: {e}")

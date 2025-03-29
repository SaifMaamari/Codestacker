#  CityX Crime Watch: Operation Safe Streets

Welcome to **CityX Crime Watch**, your data-powered solution to combat rising crime in a fictional city called CityX. This app is developed as part of the **2025 ML Rihal Codestacker Challenge** and uses machine learning and interactive geo-visualizations to empower the police with actionable insights.

##  What's Inside?

This Streamlit-based application has multiple crime intelligence tools integrated into a single interface:

###  1. Crime Data Exploration & Visualization
- Filter crime reports by date, category, and police district.
- Interactive **heatmaps** and **marker maps** of crime hotspots.
- Toggle heatmap on/off for better location clarity.
- Time-based trends visualized using `seaborn`.

###  2. Crime Category & Severity Prediction
- Enter a textual crime description to **predict the category**.
- Crimes are automatically assigned a **severity level (1–5)** based on their impact.

###  3. Police Report PDF Extraction
- Upload scanned or typed PDF reports.
- Text is extracted using **PyMuPDF** and sent through the trained model.
- Predicted crime category + severity displayed instantly.

---

##  How to Run This App

###  **Using Docker**

Make sure Docker is installed, then build and run the container:

```bash
docker build -t cityx-crime-watch .
docker run -p 8501:8501 cityx-crime-watch

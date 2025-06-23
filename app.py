import streamlit as st
import pandas as pd
import torch
from torch_geometric.data import Data
from config import DATA_PATH, MODEL_PATH
from utils.preprocessing import load_data, normalize_addresses, cluster_geolocations, analyze_traffic_patterns
from utils.model import DynamicRouteOptimizer
import folium
from streamlit_folium import folium_static

st.title('Dynamic Route Optimization')

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader('Upload route data CSV', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.sidebar.write('Using sample data')
    df = load_data(DATA_PATH)

# Preprocessing steps
df = normalize_addresses(df)
df = cluster_geolocations(df)
df = analyze_traffic_patterns(df)

st.header('Data Preview')
st.dataframe(df[['address','latitude','longitude','travel_time','cluster']].rename(
    columns={'travel_time':'original_time'}
).style.format({
    'latitude':'{:.4f}', 
    'longitude':'{:.4f}', 
    'original_time':'{:.2f}'
}))

# Load or initialize model
if 'optimizer' not in st.session_state:
    optimizer = DynamicRouteOptimizer()
    try:
        optimizer.load(MODEL_PATH)
        st.sidebar.success('Loaded existing model.')
    except FileNotFoundError:
        st.sidebar.warning('No pre-trained model found; please train.')
    st.session_state.optimizer = optimizer

optimizer = st.session_state.optimizer

# Train model button
if st.sidebar.button('Train Model'):
    data = Data(
        x=torch.tensor(df[['latitude','longitude','cluster']].values, dtype=torch.float),
        edge_index=torch.tensor([[0],[1]], dtype=torch.long),
        y=torch.tensor(df['travel_time'].values, dtype=torch.float)
    )
    optimizer.train(data)
    optimizer.save(MODEL_PATH)
    st.sidebar.success('Model trained and saved')

# Optimize route button with enhanced results
if st.sidebar.button('Optimize Route'):
    # Prepare graph data
    data = Data(
        x=torch.tensor(df[['latitude','longitude','cluster']].values, dtype=torch.float),
        edge_index=torch.tensor([[0],[1]], dtype=torch.long)
    )
    pred = optimizer.predict(data).view(-1).numpy()

    # Combine into DataFrame
    results = pd.DataFrame({
        'address': df['address'],
        'original_time': df['travel_time'].round(2),
        'optimized_time': pred.round(2)
    })

    # Display formatted results table
    st.header('Optimization Results')
    st.dataframe(results.style.format({
        'original_time':'{:.2f}',
        'optimized_time':'{:.2f}'
    }))

    # Map with popups showing both times
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for idx, row in df.iterrows():
        orig = results.loc[idx, 'original_time']
        opt = results.loc[idx, 'optimized_time']
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"{row['address']}\nOriginal: {orig:.2f}m\nOptimized: {opt:.2f}m"
        ).add_to(m)
    folium_static(m)

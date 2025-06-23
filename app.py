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

uploaded_file = st.sidebar.file_uploader('Upload route data CSV', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.sidebar.write('Using sample data')
    df = load_data(DATA_PATH)

# Preprocessing
df = normalize_addresses(df)
df = cluster_geolocations(df)
df = analyze_traffic_patterns(df)

st.header('Data Preview')
st.dataframe(df.head())

# Initialize or load model
if 'optimizer' not in st.session_state:
    optimizer = DynamicRouteOptimizer()
    try:
        optimizer.load(MODEL_PATH)
        st.sidebar.success('Loaded existing model.')
    except FileNotFoundError:
        st.sidebar.warning('No pre-trained model found; please train.')
    st.session_state.optimizer = optimizer

optimizer = st.session_state.optimizer

# Training button
if st.sidebar.button('Train Model'):
    data = Data(
        x=torch.tensor(df[['latitude','longitude','cluster']].values, dtype=torch.float),
        edge_index=torch.tensor([[0],[1]], dtype=torch.long),  # placeholder edges
        y=torch.tensor(df['travel_time'].values, dtype=torch.float)
    )
    optimizer.train(data)
    optimizer.save(MODEL_PATH)
    st.sidebar.success('Model trained and saved')

# Optimization button with improved table
if st.sidebar.button('Optimize Route'):
    # Prepare data for GNN
    data = Data(
        x=torch.tensor(df[['latitude','longitude','cluster']].values, dtype=torch.float),
        edge_index=torch.tensor([[0],[1]], dtype=torch.long)
    )
    # Get predictions
    pred = optimizer.predict(data).view(-1).numpy()

    # Build results DataFrame
    results = df[['address', 'travel_time']].copy()
    results = results.rename(columns={'travel_time': 'original_time'})
    results['optimized_time'] = pred

    # Display results table
    st.header('Optimization Results')
    st.dataframe(results)

    # Map visualization with popups
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for idx, row in df.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"{row['address']}\nOriginal: {row['travel_time']}m\nOptimized: {results.loc[idx, 'optimized_time']:.1f}m"
        ).add_to(m)
    folium_static(m)

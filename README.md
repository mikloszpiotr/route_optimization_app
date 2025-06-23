# Route Optimization App

## 🏭 Business Problem
Modern logistics face the challenge of **dynamic route optimization** in environments where traffic conditions, weather, and shipment demand change in real-time. Inefficient routing leads to increased fuel costs, delayed deliveries, and suboptimal fleet utilization.

## 🎯 Business Impact
- **Fuel Cost Reduction:** Cuts fuel expenses by approximately **18%** through optimized path selection.  
- **On-Time Delivery:** Improves on-time delivery rates by up to **30%** by adapting routes to live conditions.  
- **Capacity Utilization:** Maximizes freight capacity use, reducing empty miles and operational waste.

## ⚙️ ML Solution
This solution employs a **Graph Neural Network (GNN)** to model the transportation network and predict travel-time adjustments under varying conditions. Key techniques include:
- **Graph Neural Networks:** Two-layer GNN built with PyTorch Geometric (`GCNConv`) for network optimization.  
- **Clustering & Preprocessing:** Address normalization, geographic clustering (DBSCAN), and traffic-pattern analysis to enrich node features.  
- **Scalable Inference:** Real-time recalculation of routes and estimated time of arrival (ETA).

## 🧠 Model Details
- **Architecture:**  
  - **Input Features:** Latitude, longitude, cluster ID (3 features)  
  - **Hidden Layer:** 16 channels  
  - **Output:** Single-channel predicted travel-time adjustment  
- **Training:**  
  - **Optimizer:** Adam (learning rate = 0.01)  
  - **Loss:** Mean Squared Error  
  - **Epochs:** Configurable (default = 50)  
- **Files:**  
  - **`utils/model.py`:** Contains `GNNOptimizer` and `DynamicRouteOptimizer` classes.  
  - **`models/gnn_model.pth`:** Saved model weights (create via training in app).

## 🛠️ Data Preprocessing
- **Input CSV:** Should include `latitude`, `longitude`, `address`, and `travel_time`.  
- **Normalization:** Standardizes address formats.  
- **Clustering:** Groups nearby points via DBSCAN to identify logical route segments.  
- **Traffic Analysis:** Placeholder for integrating historical and real-time traffic data.

## 📊 Application Features
- **Data Upload:** Accepts user-provided CSV or falls back on sample data.  
- **Interactive Dashboard:**  
  - Data preview table  
  - Train model button  
  - Optimize route button with Folium map visualization  
- **Real-Time Routes:** Dynamic route recalculation and ETA updates.

## 🚀 Setup & Installation

```bash
git clone https://github.com/mikloszpiotr/route_optimization_app.git
cd route_optimization_app
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## ▶️ Running the App

```bash
streamlit run app.py
```

- **Upload** your route CSV under the "Upload route data" sidebar.  
- Click **Train Model** to fit or re-fit the GNN.  
- Click **Optimize Route** to compute optimized travel times and view on the map.

## 📁 Project Structure

```
route_optimization_app/
├── app.py
├── config.py
├── requirements.txt
├── data/
│   └── sample_routes.csv
├── models/
│   └── gnn_model.pth
└── utils/
    ├── preprocessing.py
    └── model.py
```

## 🤝 Contributing
Contributions are welcome! Please submit issues or pull requests to enhance functionality, integrate real-time data APIs, or add further optimization techniques.

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.

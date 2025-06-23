# Route Optimization App

A Streamlit-based dynamic route optimization app leveraging graph neural networks to minimize fuel costs and boost on-time deliveries.

## 🚀 Features

- **GNN-powered routing** using PyTorch Geometric  
- Dynamic traffic, weather, and demand-aware path optimization  
- Real-time map visualization with Folium  
- One-click training & inference in Streamlit  

## 🛠️ Setup

```bash
git clone https://github.com/mikloszpiotr/route_optimization_app.git
cd route_optimization_app
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run app.py
```

## 📊 Data & Models

- Drop your route CSV into `data/sample_routes.csv`  
- Trained model checkpoints go into `models/gnn_model.pth`

## 🤝 Contributing

Feel free to open issues or pull requests!

# 📦 PriceOptima – AI-Based Dynamic Pricing System

## 🎯 Project Objective
PriceOptima is an AI-driven dynamic pricing system designed to improve revenue by recommending optimal product prices based on demand patterns, inventory levels, and time-based factors.  
The project moves beyond static pricing by using machine learning models to simulate real-world retail pricing decisions and validate revenue improvement through historical backtesting.

---

## 📊 Dataset Description
The dataset contains historical retail transaction data, including:
- Product prices  
- Units sold (demand)  
- Inventory levels  
- Date and time attributes  

Additional features such as **day of week**, **month**, and **weekend indicators** were engineered to capture demand trends.  
The dataset was cleaned by removing duplicates, handling missing values, and preparing it for modeling.

---

## 🛠️ Technologies Used
- **Languages:** Python, JavaScript  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, LightGBM, XGBoost  
- **Backend API:** FastAPI  
- **Frontend Dashboard:** React.js  
- **Model Serialization:** Joblib  
- **API Server:** Uvicorn  

---

## 🤖 Model Development Summary
- Feature engineering was applied to generate meaningful inputs for demand prediction.
- A **time-based train-test split** was used to simulate real-world scenarios.
- Two models were trained and evaluated:
  - XGBoost Regressor  
  - LightGBM Regressor  
- Models were evaluated using **MAE** and **RMSE** metrics.
- **LightGBM** performed better and was selected for deployment.
- Predicted demand values were used to simulate ML-based pricing decisions.

![alt text](<Screenshot 2025-12-28 102815.png>)

---

## 🚀 Backend Implementation (FastAPI)
- The trained LightGBM model was saved using `joblib`.
- A FastAPI backend was created to load the model and expose pricing functionality via REST APIs.
- The `/predict-price` endpoint accepts pricing-related inputs and returns:
  - Predicted demand  
  - Recommended dynamic price  
- Swagger UI (`/docs`) was used to test and validate API functionality.

![alt text](<Screenshot 2025-12-28 102854.png>)

---

## 🖥️ Dashboard Implementation (React.js)
- A React.js dashboard was developed for user interaction.
- Features include:
  - Input fields for pricing parameters  
  - Clear labels and example placeholders  
  - Loading indicator during prediction  
  - Display of predicted demand and recommended price  
- The dashboard communicates with the FastAPI backend using REST API calls.

![alt text](<Screenshot 2026-01-05 211618.png>)

---

## ⚙️ Deployment & Execution Approach

 - The backend API is developed using FastAPI and runs locally via Uvicorn.

 - The backend server is accessible at:
  -  http://127.0.0.1:8000

 - API endpoints were tested and verified using Swagger UI available at:
  - http://127.0.0.1:8000/docs

 - All backend dependencies are managed through the requirements.txt file.

 - The frontend dashboard is built using React and executed locally using npm start.

 - The frontend application runs successfully at:
  - http://localhost:3000

 - Communication between the React frontend and FastAPI backend was validated through successful API calls.

---

## 📈 Key Outputs & Results
The **ML-based pricing strategy achieved a positive revenue lift** compared to static pricing, validating the effectiveness of demand-driven dynamic pricing.

![alt text](<Screenshot 2026-01-05 212527.png>)

---
## 🌍 Real-World Scenarios
The PriceOptima system can be used by retail businesses and e-commerce platforms to make data-driven pricing decisions in real time. Instead of using fixed prices, the system dynamically adjusts product prices based on predicted customer demand, inventory levels, and time-based factors such as weekends or seasonal trends.

---
## 🔮 Conclusion and Future Enhancements
This project demonstrates a complete end-to-end AI-powered pricing system using machine learning, backend APIs, and a frontend dashboard.

**Future enhancements may include:**
- Real-time demand forecasting  
- Additional customer and market features  
- Advanced pricing optimization strategies  
- Cloud deployment and scalability improvements  

---

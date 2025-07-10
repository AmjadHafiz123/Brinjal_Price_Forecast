# 🥦 Vegetable Price Forecast API

This is a Flask API to forecast vegetable prices using a trained Prophet model. The model supports holiday effects and returns price predictions along with trend and confidence levels.

---

## 📁 Project Structure

vegetable-forecast-api/
│
├── app.py # Main Flask application
├── brinjal_model.pkl # Trained Prophet model for brinjal
├── scaler.pkl # Scaler used for inverse transformation
├── requirements.txt # Python package dependencies
├── README.md # This file
└── venv/ # (Optional) Virtual environment

yaml
Copy
Edit

---

## ⚙️ Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/)
- [VS Code](https://code.visualstudio.com/) or any IDE

---

## 🧪 Setup Instructions

### 1. Clone or Download the Project

```bash
git clone https://github.com/AmjadHafiz123/Brinjal_Price_Forecast.git
cd vegetable-forecast-api


2. Create and Activate a Virtual Environment
python -m venv venv
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt
If prophet fails to install, try:


pip install prophet
Or for legacy versions:


pip install fbprophet
🚀 Running the Flask Server
Set the Flask app and run:

export FLASK_APP=app
flask run
For Windows CMD:

set FLASK_APP=app
flask run
Server will be available at:
http://127.0.0.1:5000

📡 API Endpoint
POST /forecast
➤ Request Body (JSON)

{
  "vegetable": "brinjal",
  "startdate": "2025-07-01",
  "enddate": "2025-07-10"
}
➤ Response Format

[
  {
    "date": "2025-07-01",
    "price": 23.45,
    "trend": "up",
    "confidence": 0.87
  },
  {
    "date": "2025-07-02",
    "price": 23.35,
    "trend": "down",
    "confidence": 0.91
  }
]

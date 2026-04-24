# 📱 WhatsApp ML Analytics Dashboard

An end-to-end machine learning project that analyzes WhatsApp chat exports, detects sentiment & abusive language, and predicts message reply times using regression models.

## 🚀 Features

- **Upload & Parse** – Upload WhatsApp `.txt` chat exports, automatic preprocessing.
- **User Analytics** – Messages, words, media, links, average message length per user.
- **Timeline & Activity** – Monthly message trends, weekday & month-wise activity heatmaps.
- **Sentiment Analysis** – Classifies messages as Positive / Negative / Neutral using TextBlob/VADER.
- **Abuse Detection** – Identifies toxic words, calculates toxicity percentage, and shows most abusive users.
- **ML Reply Prediction** – Trains a regression model (Random Forest / XGBoost) to predict reply time (minutes) based on:
  - Word count of the message  
  - Hour of day  
  - Sender  
  - Day of week  

## 🧠 Machine Learning Components

| Component | Type | Libraries |
|-----------|------|------------|
| Sentiment Analysis | Rule‑based / Lexicon | TextBlob / NLTK |
| Abuse Detection | Keyword matching + optional transformer | Custom wordlist / detoxify |
| Reply Time Prediction | Regression (Random Forest) | scikit‑learn, pandas, numpy |

**Reply Time Engineering**  
For each message, reply time = minutes until the *same user* sends the next message.  
Features: `word_count`, `hour`, `sender_encoded`, `day_encoded`.  
Evaluation metrics: MAE, RMSE, R².

## 📸 Screenshots

*(Add 2-3 screenshots of the dashboard after running)*

## 🛠️ Tech Stack

- **Frontend & Dashboard**: Streamlit  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Plotly Express  
- **ML & NLP**: scikit‑learn, TextBlob, NLTK  
- **Model Persistence**: joblib (optional)

## 📂 Project Structure

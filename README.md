# 📱 WhatsApp ML Analytics Dashboard

An end-to-end machine learning project that analyzes WhatsApp chat exports, detects sentiment & abusive language, and predicts message reply times using regression models.

Demo: https://whatsapp-chat-analyzer-dfn7sj7rq77iazfmmaqkhi.streamlit.app/

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
WhatsApp-ML-Dashboard/
│
├── app.py # Main Streamlit application
├── preprocessor.py # Chat parsing & feature extraction
├── helper.py # Statistics, sentiment, abuse functions
├── ml_reply_model.py # Reply time model training & prediction
├── requirements.txt # Python dependencies
├── README.md # This file
└── sample_chat.txt # (Optional) sample WhatsApp export

text

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/hitesh-11-lab/WhatsApp-ML-Dashboard.git
cd WhatsApp-ML-Dashboard
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the dashboard
bash
streamlit run app.py
4. Use the app
Upload a WhatsApp chat export (.txt file)

Select a user or “Overall”

Click Analyze to see statistics and visualizations

Go to Reply Prediction tab → click Train Reply Model → then predict reply time for any message

📄 Expected Chat Format
The app expects a standard WhatsApp chat export (without media path). Example:

text
12/10/23, 10:15 AM - Alice: Hey, how are you?
12/10/23, 10:17 AM - Bob: I'm good, thanks! What about you?
12/10/23, 10:20 AM - Alice: Same here. Let's meet tomorrow.
Group notifications (e.g., “Alice created group”) are automatically filtered out.

📊 Sample Results
Overview: Total messages, words, media shared, links, average length.

Activity: Busiest day of week and month.

Sentiment: Pie chart of overall sentiment distribution.

Abuse: Top abusive words and user-wise table.

Reply Prediction: Trains model with MAE ~X minutes, RMSE ~Y minutes, R² ~Z.

🔧 Customization
Change ML model – Edit ml_reply_model.py to use XGBoost, SVM, or neural networks.

Add new features – Extend preprocessor.py to extract emoji counts, message length, etc.

Improve abuse detection – Integrate detoxify or a fine‑tuned BERT model.

📌 Future Improvements
Word cloud for most frequent words per user

Emoji analysis & timeline

Export reports as PDF

Deploy on Streamlit Cloud for online access

👥 Contributors
Your Name – @yourusername

(Add teammate names if any)

📝 License
This project is for educational purposes as part of a college mini project.

🙏 Acknowledgements
WhatsApp for chat export feature

Streamlit, Plotly, scikit‑learn open‑source communities

NLTK / TextBlob for sentiment lexicons

🎓 Submitted for: [Course Name, College Name]
📅 Date: [Submission Date]

text

## ✅ Instructions for You

1. **Replace placeholders** – Your name, GitHub username, course name, college name, date.
2. **Add 2-3 screenshots** – In a `screenshots/` folder, then link them in the README.
3. **Create a sample chat file** – Save `sample_chat.txt` (anonymized) in the repo.
4. **Ensure `requirements.txt` exists** – Include all libraries you imported.

If you want me to generate `requirements.txt` based on typical imports (streamlit, pandas, plotly, scikit-learn, textblob, nltk), just say so!


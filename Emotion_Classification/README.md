
# 🎭 Emotion Analyzer Web App

This project is a web-based **Emotion Analyzer** built with **FastAPI**, allowing users to upload text datasets (CSV or Excel), perform emotion classification, visualize results, and analyze sentiments using **Google Gemini**. The app supports advanced text processing and generates insightful visualizations like word clouds and emotion distributions.

---

## 🚀 Features

- 📁 Upload CSV or Excel files
- 🔍 Select and clean a text column for analysis
- 🧠 Classify emotions using a HuggingFace BERT pipeline
- 📊 Generate emotion distribution charts using Matplotlib + Seaborn
- ☁️ Create word clouds for each detected emotion
- 🧠 Use **Google Gemini (Generative AI)** to summarize insights from word clouds
- 📂 Serve static visual outputs from the `/static` directory

---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Emotion_Classification/static/T1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Emotion_Classification/static/T2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Emotion_Classification/static/T3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Emotion_Classification/static/T4.JPG)


## 🛠️ Tech Stack & Tools

| Category        | Library / Tool                           |
|----------------|-------------------------------------------|
| Web Framework   | FastAPI                                  |
| Data Handling   | Pandas, Polars                           |
| Visualization   | Matplotlib, Seaborn, WordCloud           |
| Emotion Model   | Transformers (`BERT-Emotions-Classifier`)|
| Image Handling  | PIL                                       |
| AI Analysis     | Google Generative AI (Gemini)            |
| Environment     | Python `dotenv` for secure API keys      |
| Logging         | Python `logging` module                  |

---

## 📁 Project Structure

```
emotion-analyzer/
├── main.py                # FastAPI app
├── static/                # Folder for generated visualizations
├── .env                   # Gemini API key (not committed)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧪 Installation

1. **Clone the repo**
```bash
git clone https://github.com/your-username/emotion-analyzer.git
cd emotion-analyzer
```

2. **Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set your Gemini API key**
Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

---

## ▶️ Run the App

```bash
uvicorn main:app --reload --port 9000
```

Then open [http://localhost:9000](http://localhost:9000) in your browser.

---

## 📊 Output Artifacts

- `static/emotion_distribution.png`: Bar chart of detected emotions
- `static/wordcloud_<emotion>.png`: Word cloud for each emotion
- Gemini-generated text insights based on word cloud tokens

---

## 🤖 Google Gemini Usage

After generating the word clouds, the app sends keywords to Google Gemini to get sentiment-based summaries. This adds a layer of NLP-driven context to visualizations by analyzing the tone and meaning of clustered keywords.

---

## 📌 TODO / Future Improvements

- Export results as PDF or downloadable reports
- Add interactive chart features (e.g., Plotly)
- Support multilingual sentiment classification
- Allow multiple file uploads or batch analysis

---

## 📝 License

MIT License

---

## 🙌 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/ayoubkirouane/BERT-Emotions-Classifier)
- [Google Generative AI](https://ai.google.dev/)

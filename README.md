# 📄 Intelligent Research Paper Summarizer with Novelty Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-Educational-green)

## 🔍 Overview

A Flask-based web application that analyzes research papers, generates AI-powered structured summaries, and measures novelty by comparing the uploaded document against live scholarly literature from multiple academic APIs.

Supports **PDF and DOCX** uploads. Models are lazy-loaded and API results are cached for faster repeated analysis.

---

## 🚀 Features

- 📑 Upload **PDF or DOCX** research papers (up to 20MB)
- 🧠 AI-based **structured summarization** (DistilBART)
- 🧩 Automatic section extraction:
  - Objective, Methodology, Results, Conclusion
- 🌐 Live paper fetching from 4 academic sources (concurrently):
  - Semantic Scholar, arXiv, OpenAlex, CORE
- 📊 **Weighted novelty score** using sentence embeddings (MiniLM)
- 🔑 Novel keyword extraction via **TF-IDF**
- 📈 Interactive similarity chart (Chart.js)
- ⬇️ Download analysis summary as a text file
- ⚡ API result caching to avoid redundant network calls

---

## 🗂️ Project Structure

```
├── static/
│   ├── script.js          # Upload page JS (drag-drop, loading state)
│   ├── style.css          # Upload page styles
│   ├── result_script.js   # Result page JS (chart, collapsibles, download)
│   └── result_style.css   # Result page styles
├── templates/
│   ├── index.html         # File upload page
│   └── result.html        # Analysis results page
├── uploads/               # Uploaded files (git-ignored)
├── models/                # Cached model weights (git-ignored)
├── app.py                 # Main Flask application
├── build_corpus.py        # Model pre-warming script
├── requirements.txt       # Python dependencies
├── .env                   # API keys (git-ignored, never commit)
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Summarization | Hugging Face Transformers (DistilBART) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Similarity | Scikit-learn TF-IDF, Cosine Similarity |
| Text Processing | NLTK, pdfplumber, python-docx |
| API Scraping | Requests, BeautifulSoup |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jeevanandan19/INTELLIGENT-RESEARCH-PAPER-SUMMARIZER-WITH-NOVELTY-DETECTION.git
cd INTELLIGENT-RESEARCH-PAPER-SUMMARIZER-WITH-NOVELTY-DETECTION
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
CORE_API_KEY=your_core_api_key_here
```
Get your free API key at [core.ac.uk/services/api](https://core.ac.uk/services/api).

> `.env` is listed in `.gitignore` — never commit it.

### 5. (Optional) Pre-warm Models
Run this once to download and cache models before the first request:
```bash
python build_corpus.py
```

### 6. Run the Application
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 📊 Novelty Score Interpretation

| Score | Label |
|---|---|
| 75% – 100% | 🌟 Highly Novel |
| 50% – 74% | 🧠 Moderately Novel |
| 0% – 49% | 📄 Low Novelty |

The score is computed as `1 − max_weighted_similarity`, where each source is weighted by credibility (CORE > Semantic Scholar > OpenAlex > arXiv).

---

## 📌 Use Cases

- Final-year / capstone project evaluation
- Research paper originality analysis
- Literature survey assistance
- Conference & journal paper pre-screening

---

## 🔮 Future Enhancements

- Plagiarism percentage detection
- LaTeX file support
- Citation network analysis
- User authentication & analysis history
- Domain-specific fine-tuned models

---

## 👤 Author

**Jeevanandan V**  
B.E. Student | AI & NLP Enthusiast

---

## 📜 License

This project is developed for **educational and research purposes**.

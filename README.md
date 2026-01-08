# 📄 Novelty Detection System for Research Papers

## 🔍 Project Overview
The **Novelty Detection System** is a Flask-based web application designed to **analyze research papers**, generate **AI-powered summaries**, and **measure novelty** by comparing the uploaded document with existing scholarly literature.

The system supports **PDF and DOCX files**, extracts structured information, fetches related papers from multiple academic sources, and computes a **novelty score** using semantic similarity techniques.

---

## 🗂️ Project Structure

```
INTELLIGENT RESEARCH PAPER SUMMARIZER WITH NOVELTY DETECTION/
│
│
├── models/                # Saved or downloaded ML/NLP models
│
├── static/                # Frontend static assets
│   ├── script.js          # Common JavaScript logic
│   ├── style.css          # Main CSS styling
│   ├── result_script.js   # Result page JS (charts & interactions)
│   └── result_style.css   # Result page CSS
│
├── templates/             # HTML templates
│   ├── index.html         # File upload page
│   └── result.html        # Results & novelty display page
│
├── uploads/               # Uploaded research papers
│
├── .env                   # Environment variables (API keys)
├── app.py                 # Main Flask application
├── build_corpus.py        # Script to build local text corpus
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Features
- 📑 Upload **PDF or DOCX** research papers
- 🧠 AI-based **automatic summarization**
- 🧩 Structured extraction:
  - Objective
  - Methodology
  - Results
  - Conclusion
- 🌐 Online paper fetching from:
  - Semantic Scholar
  - arXiv
  - OpenAlex
  - CORE
- 📊 **Novelty score computation** using sentence embeddings
- 🔑 Novel keyword extraction using **TF-IDF**
- 📈 Similarity visualization on results page

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask

### NLP & Machine Learning
- Hugging Face Transformers (DistilBART)
- Sentence Transformers (MiniLM)
- Scikit-learn (TF-IDF)
- NLTK

### Data Processing
- pdfplumber
- python-docx
- BeautifulSoup
- Requests

### Frontend
- HTML5
- CSS3
- JavaScript

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jeevanandan19/INTELLIGENT-RESEARCH-PAPER-SUMMARIZER-WITH-NOVELTY-DETECTION.git
cd novelty-detection
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file:
```env
CORE_API_KEY=zj1gPHfwsAEcN6ql2JkoLCDOZQt5vYSp
```

### 5️⃣ Run the Application
```bash
python app.py
```

Access the application at:
```
http://127.0.0.1:5000
```

---

## 📊 Novelty Score Interpretation
| Score Range | Interpretation      |
|------------ |---------------------|
| ≥ 75%       | 🌟 Highly Novel     |
| 50% – 74%   | 🧠 Moderately Novel |
| < 50%       | 📄 Low Novelty      |

---

## 📌 Use Cases
- Final-year engineering projects
- Research paper originality analysis
- Literature survey assistance
- Conference & journal paper screening

---

## 🔮 Future Enhancements
- Plagiarism percentage detection
- Support for LaTeX files
- Citation network analysis
- User authentication & history tracking
- Domain-specific fine-tuned models

---

## 👤 Author
**Jeevanandan V**  
B.E. Student | AI & NLP Enthusiast

---

## 📜 License
This project is developed for **educational and research purposes**.

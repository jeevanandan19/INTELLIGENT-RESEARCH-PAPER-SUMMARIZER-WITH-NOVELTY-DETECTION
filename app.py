import docx
import os
import re
import requests
import pdfplumber
import nltk
import hashlib
import json
from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

# Weight assigned to each paper source (higher = more credible)
SOURCE_WEIGHTS = {
    "CORE": 1.0,
    "Semantic Scholar": 0.9,
    "OpenAlex": 0.8,
    "arXiv": 0.7,
    "Unknown Source": 0.5
}

# Simple in-memory cache for API results {query_hash: papers_list}
_api_cache = {}

# Ensure nltk tokenizer data is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MAX_UPLOAD_MB = 20
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024  # 20 MB limit

# ---------------------------------------
# Lazy-loaded models (loaded once on first use)
# ---------------------------------------
_summarizer = None
_embedder = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder="models"
        )
    return _embedder

# ---------------------------------------
# Utility: PDF text extraction
# ---------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text.strip()

# ---------------------------------------
# General file text extractor (PDF or DOCX)
# ---------------------------------------
def extract_text_from_file(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
        except Exception as e:
            print(f"DOCX extraction failed: {e}")
            return ""
    else:
        print("Unsupported file format.")
        return ""

# ---------------------------------------
# Novelty keyword extraction
# ---------------------------------------
def extract_novelty_words(input_text, similar_texts, top_n=10):
    corpus = [input_text] + similar_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = X.toarray()[0]
        sorted_indices = tfidf_scores.argsort()[::-1][:top_n]
        return [feature_names[i] for i in sorted_indices]
    except Exception:
        return []

# ---------------------------------------
# Structured summarization
# ---------------------------------------
def structured_summary(text):
    """Generate structured summary for research papers."""
    summarizer = get_summarizer()
    sents = nltk.sent_tokenize(text)
    if not sents:
        return "No readable content extracted from file.", {}

    # Split into safe chunks (~700 words each)
    chunks, current_chunk, count = [], [], 0
    for sent in sents:
        words = sent.split()
        count += len(words)
        current_chunk.append(sent)
        if count > 700:
            chunks.append(" ".join(current_chunk))
            current_chunk, count = [], 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            words = chunk.split()
            if len(words) > 800:
                chunk = " ".join(words[:800])
            result = summarizer(chunk, max_length=180, min_length=40, do_sample=False, truncation=True)
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            print(f"Summarization failed for chunk {i}: {e}")
            summaries.append(" ".join(chunk.split()[:100]))

    combined_summary = " ".join(summaries).strip() or "Automatic summarization failed. Please upload a cleaner file."

    # Extract structured sections
    text_lower = text.lower()
    sections = {"Objective": "", "Method": "", "Results": "", "Conclusion": ""}
    patterns = {
        "Objective": r"(objective|aim|goal|purpose|motivation)[\s\S]{0,800}",
        "Method": r"(method|approach|architecture|technique|proposed system)[\s\S]{0,800}",
        "Results": r"(result|performance|evaluation|accuracy|experiment)[\s\S]{0,800}",
        "Conclusion": r"(conclusion|in summary|in conclusion|we conclude)[\s\S]{0,800}"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            start = match.start()
            snippet = text[start:start + 700]
            try:
                sec_sum = summarizer(snippet, max_length=100, min_length=30, truncation=True)[0]["summary_text"]
                sections[key] = sec_sum
            except Exception as e:
                print(f"Section summarization failed for {key}: {e}")
                sections[key] = snippet[:200]

    structured = f"""OBJECTIVE: {sections['Objective'] or 'Not clearly stated.'}
METHOD: {sections['Method'] or 'Not described.'}
RESULT: {sections['Results'] or 'No results mentioned.'}
CONCLUSION: {sections['Conclusion'] or 'No clear conclusion found.'}"""

    return structured.strip(), sections

# ---------------------------------------
# Fetch from APIs
# ---------------------------------------
def fetch_semantic_scholar(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,abstract,url"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [
            {"title": p.get("title", ""), "abstract": p.get("abstract", ""),
             "source": "Semantic Scholar", "url": p.get("url", "#")}
            for p in data if p.get("abstract")
        ]
    except Exception as e:
        print("Semantic Scholar error:", e)
        return []

def fetch_arxiv(query, limit=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        return [
            {"title": e.title.text.strip(), "abstract": e.summary.text.strip(),
             "source": "arXiv", "url": e.id.text.strip()}
            for e in soup.find_all("entry")
        ]
    except Exception as e:
        print("arXiv error:", e)
        return []

def fetch_openalex(query, limit=5):
    url = f"https://api.openalex.org/works?filter=title.search:{query}&per-page={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        papers = []
        for p in data:
            abstract_dict = p.get("abstract_inverted_index")
            abstract = " ".join(abstract_dict.keys()) if abstract_dict else ""
            papers.append({
                "title": p.get("display_name", ""),
                "abstract": abstract,
                "source": "OpenAlex",
                "url": p.get("id", "#")
            })
        return papers
    except Exception as e:
        print("OpenAlex error:", e)
        return []

def fetch_core(query, limit=5):
    api_key = os.getenv("CORE_API_KEY")
    if not api_key:
        return []
    url = f"https://api.core.ac.uk/v3/search/works?limit={limit}&q={query}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        return [
            {"title": p.get("title", "Untitled"), "abstract": p.get("abstract", ""),
             "source": "CORE", "url": p.get("downloadUrl", "#")}
            for p in data
        ]
    except Exception as e:
        print("CORE API error:", e)
        return []

def fetch_all_sources(query):
    """Fetch from all 4 APIs concurrently with caching."""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    if query_hash in _api_cache:
        print("Using cached API results.")
        return _api_cache[query_hash]

    results = []
    fetchers = [fetch_semantic_scholar, fetch_arxiv, fetch_openalex, fetch_core]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn, query, 5): fn.__name__ for fn in fetchers}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"Fetch error ({futures[future]}):", e)

    _api_cache[query_hash] = results
    return results

# ---------------------------------------
# Novelty Analysis
# ---------------------------------------
def online_novelty_analysis(input_text, query_terms):
    embedder = get_embedder()
    papers = fetch_all_sources(query_terms)
    if not papers:
        return 1.0, [], {"total_papers": 0, "avg_similarity": 0.0, "max_similarity": 0.0}

    abstracts = [p["abstract"] for p in papers if p.get("abstract", "").strip()]
    if not abstracts:
        return 1.0, papers, {"total_papers": len(papers), "avg_similarity": 0.0, "max_similarity": 0.0}

    online_embs = embedder.encode(abstracts, convert_to_tensor=True, batch_size=32)
    input_emb = embedder.encode([input_text], convert_to_tensor=True)
    sims = util.cos_sim(input_emb, online_embs)[0].cpu().tolist()

    weighted_sims = []
    for i, sim in enumerate(sims):
        source = papers[i].get("source", "Unknown Source")
        weight = SOURCE_WEIGHTS.get(source, 0.5)
        weighted_sim = sim * weight
        weighted_sims.append(weighted_sim)
        papers[i]["similarity"] = round(sim, 3)
        papers[i]["weighted_similarity"] = round(weighted_sim, 3)

    max_sim = max(weighted_sims)
    avg_sim = sum(weighted_sims) / len(weighted_sims)
    novelty_score = round(1 - max_sim, 3)

    metrics = {
        "total_papers": len(papers),
        "avg_similarity": round(avg_sim, 3),
        "max_similarity": round(max_sim, 3)
    }

    # Sort by similarity descending, return top 10
    papers_with_abstract = [p for p in papers if p.get("abstract")]
    papers_with_abstract.sort(key=lambda x: x.get("weighted_similarity", 0), reverse=True)
    return novelty_score, papers_with_abstract[:10], metrics

# ---------------------------------------
# Flask Routes
# ---------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("paper")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select a file.")
        if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".docx")):
            return render_template("index.html", error="Only PDF or DOCX files are allowed.")
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        return redirect(url_for("analyze", filename=file.filename))
    return render_template("index.html")

@app.route("/analyze/<filename>")
def analyze(filename):
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(pdf_path):
        return render_template("index.html", error="File not found. Please upload again."), 404

    text = extract_text_from_file(pdf_path)
    if not text:
        return render_template("index.html", error="Could not extract text from the file. Try a different file.")

    summary, sections = structured_summary(text)
    keywords = " ".join(summary.split()[:10])
    novelty_score, similar_papers, metrics = online_novelty_analysis(summary, keywords)

    if novelty_score >= 0.75:
        novelty_label = "🌟 Highly Novel"
    elif novelty_score >= 0.5:
        novelty_label = "🧠 Moderately Novel"
    else:
        novelty_label = "📄 Low Novelty"

    safe_papers = [
        {
            "title": p.get("title") or "Untitled Paper",
            "abstract": p.get("abstract") or "No abstract available.",
            "source": p.get("source") or "Unknown Source",
            "url": p.get("url") or "#",
            "similarity": float(p.get("similarity", 0.0))
        }
        for p in similar_papers
    ]

    similar_texts = [p["abstract"] for p in safe_papers if p["abstract"]]
    novelty_words = extract_novelty_words(summary, similar_texts, top_n=10)

    return render_template(
        "result.html",
        filename=filename,
        summary=summary,
        sections=sections,
        novelty_pct=round(novelty_score * 100, 2),
        novelty_label=novelty_label,
        metrics=metrics,
        similar_papers=safe_papers,
        chart_labels=[p["title"] for p in safe_papers],
        chart_values=[round(p["similarity"] * 100, 2) for p in safe_papers],
        chart_urls=[p["url"] for p in safe_papers],
        chart_sources=[p["source"] for p in safe_papers],
        novelty_words=novelty_words
    )

@app.errorhandler(413)
def file_too_large(e):
    return render_template("index.html", error=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB}MB."), 413

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

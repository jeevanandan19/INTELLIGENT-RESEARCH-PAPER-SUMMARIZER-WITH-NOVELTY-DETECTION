import docx
import os
import re
import requests
import pdfplumber
import nltk
from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv


# Weight assigned to each paper source (higher = more credible)
SOURCE_WEIGHTS = {
    "CORE": 1.0,
    "Semantic Scholar": 0.9,
    "OpenAlex": 0.8,
    "arXiv": 0.7,
    "Unknown Source": 0.5
}

load_dotenv()


def extract_novelty_words(input_text, similar_texts, top_n=10):
    corpus = [input_text] + similar_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]  # For input_text
    # Lower scores mean more common words, high = more unique
    sorted_indices = tfidf_scores.argsort()[::-1][:top_n]
    novelty_words = [feature_names[i] for i in sorted_indices]
    return novelty_words

# Ensure nltk tokenizer data is available
nltk.download("punkt")

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        # Use existing PDF extractor
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
        except Exception as e:
            print(f"⚠️ DOCX extraction failed: {e}")
            return ""
    else:
        print("⚠️ Unsupported file format.")
        return ""


# ---------------------------------------
# Structured summarization (robust)
# ---------------------------------------
def structured_summary(text):
    """Generate structured and reliable summary for research papers."""
    sents = nltk.sent_tokenize(text)
    if not sents:
        return "No readable content extracted from PDF.", {}

    # Split into small safe chunks (to avoid token overflow)
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

            result = summarizer(
                chunk,
                max_length=180,
                min_length=40,
                do_sample=False,
                truncation=True
            )
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            print(f"⚠️ Summarization failed for chunk {i}: {e}")
            fallback = " ".join(chunk.split()[:100])
            summaries.append(fallback)

    combined_summary = " ".join(summaries).strip()
    if not combined_summary:
        combined_summary = "Automatic summarization failed. Please upload a cleaner PDF."

    # Extract structured sections
    text_lower = text.lower()
    sections = {
        "Objective": "",
        "Method": "",
        "Results": "",
        "Conclusion": ""
    }

    # Keyword patterns to detect sections
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
            snippet = text[start:start+700]
            try:
                sec_sum = summarizer(snippet, max_length=100, min_length=30, truncation=True)[0]["summary_text"]
                sections[key] = sec_sum
            except Exception as e:
                print(f"⚠️ Section summarization failed for {key}: {e}")
                sections[key] = snippet[:200]

    # Create structured text
    structured_summary_text = f"""
    OBJECTIVE: {sections['Objective'] or 'Not clearly stated.'}
    METHOD: {sections['Method'] or 'Not described.'}
    RESULT: {sections['Results'] or 'No results mentioned.'}
    CONCLUSION: {sections['Conclusion'] or 'No clear conclusion found.'}
"""

    return structured_summary_text.strip(), sections

# ---------------------------------------
# Fetch from APIs
# ---------------------------------------
def fetch_semantic_scholar(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,abstract,url"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json().get("data", [])
        return [
            {
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "source": "Semantic Scholar",
                "url": p.get("url", "#")
            }
            for p in data if p.get("abstract")
        ]
    except Exception as e:
        print("Semantic Scholar error:", e)
        return []

def fetch_arxiv(query, limit=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={limit}"
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "xml")
        entries = soup.find_all("entry")
        return [
            {
                "title": e.title.text.strip(),
                "abstract": e.summary.text.strip(),
                "source": "arXiv",
                "url": e.id.text.strip()
            } for e in entries
        ]
    except Exception as e:
        print("arXiv error:", e)
        return []

def fetch_openalex(query, limit=5):
    url = f"https://api.openalex.org/works?filter=title.search:{query}&per-page={limit}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json().get("results", [])
        papers = []
        for p in data:
            title = p.get("display_name", "")
            abstract_dict = p.get("abstract_inverted_index")
            abstract = " ".join(abstract_dict.keys()) if abstract_dict else ""
            papers.append({
                "title": title,
                "abstract": abstract,
                "source": "OpenAlex",
                "url": p.get("id", "#")
            })
        return papers
    except Exception as e:
        print("OpenAlex error:", e)
        return []
def fetch_core(query, limit=5):
    api_key = os.getenv("CORE_API_KEY")  # You can store it in .env
    url = f"https://api.core.ac.uk/v3/search/works?limit={limit}&q={query}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("results", [])
            papers = []
            for p in data:
                papers.append({
                    "title": p.get("title", "Untitled"),
                    "abstract": p.get("abstract", ""),
                    "source": "CORE",
                    "url": p.get("downloadUrl", "#")
                })
            return papers
        else:
            print("CORE API error:", resp.text)
            return []
    except Exception as e:
        print("CORE API exception:", e)
        return []

# Fetch from all 4 APIs concurrently
def fetch_all_sources(query):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_semantic_scholar, query, 5),
            executor.submit(fetch_arxiv, query, 5),
            executor.submit(fetch_openalex, query, 5),
            executor.submit(fetch_core, query, 5)
        ]
        for f in futures:
            try:
                results.extend(f.result())
            except Exception as e:
                print("Fetch error:", e)
    return results

# ---------------------------------------
# Novelty Analysis
# ---------------------------------------
def online_novelty_analysis(input_text, query_terms):
    papers = fetch_all_sources(query_terms)
    if not papers:
        return 1.0, [], {
            "total_papers": 0,
            "avg_similarity": 0.0,
            "max_similarity": 0.0
        }

    #  Extract all valid abstracts
    abstracts = [p["abstract"] for p in papers if p.get("abstract", "").strip()]
    if not abstracts:
        return 1.0, papers, {
            "total_papers": len(papers),
            "avg_similarity": 0.0,
            "max_similarity": 0.0
        }

    # Encode embeddings
    online_embs = embedder.encode(abstracts, convert_to_tensor=True)
    input_emb = embedder.encode([input_text], convert_to_tensor=True)
    sims = util.cos_sim(input_emb, online_embs)[0].cpu().tolist()

    #  Attach similarity + apply weights
    weighted_sims = []
    for i, sim in enumerate(sims):
        source = papers[i].get("source", "Unknown Source")
        weight = SOURCE_WEIGHTS.get(source, 0.5)
        weighted_sim = sim * weight
        weighted_sims.append(weighted_sim)
        papers[i]["similarity"] = round(sim, 3)
        papers[i]["weighted_similarity"] = round(weighted_sim, 3)

    #  Compute weighted metrics
    max_sim = max(weighted_sims)
    avg_sim = sum(weighted_sims) / len(weighted_sims)
    novelty_score = round(1 - max_sim, 3)

    metrics = {
        "total_papers": len(papers),
        "avg_similarity": round(avg_sim, 3),
        "max_similarity": round(max_sim, 3)
    }

    return novelty_score, papers[:10], metrics

# ---------------------------------------
# Flask Routes
# ---------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("paper")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select a PDF file.")
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
        return "File not found", 404

    text = extract_text_from_file(pdf_path)
    summary, sections = structured_summary(text)
    keywords = " ".join(summary.split()[:10])
    novelty_score, similar_papers, metrics = online_novelty_analysis(summary, keywords)

    # Label novelty level
    if novelty_score >= 0.75:
        novelty_label = "🌟 Highly Novel"
    elif novelty_score >= 0.5:
        novelty_label = "🧠 Moderately Novel"
    else:
        novelty_label = "📄 Low Novelty"

    # 🧠 FIX: ensure all paper entries have safe values
    safe_papers = []
    for p in similar_papers:
        safe_papers.append({
            "title": p.get("title") or "Untitled Paper",
            "abstract": p.get("abstract") or "No abstract available.",
            "source": p.get("source") or "Unknown Source",
            "url": p.get("url") or "#",
            "similarity": float(p.get("similarity", 0.0))
        })
    similar_texts = [p["abstract"] for p in similar_papers if p["abstract"]]
    novelty_words = extract_novelty_words(summary, similar_texts, top_n=10)

    # Send data safely
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

if __name__ == "__main__":
    print("🚀 Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)


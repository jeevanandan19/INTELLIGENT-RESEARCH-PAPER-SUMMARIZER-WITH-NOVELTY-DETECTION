# build_corpus.py
# Utility to pre-warm models so the first request is faster.
# Run this once after setup: python build_corpus.py

from app import get_summarizer, get_embedder

if __name__ == "__main__":
    print("Loading summarizer model...")
    get_summarizer()
    print("Loading embedder model...")
    get_embedder()
    print("Models loaded and cached. First request will be fast.")

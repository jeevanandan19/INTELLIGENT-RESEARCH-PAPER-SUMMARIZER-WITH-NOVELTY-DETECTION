# build_corpus.py
import sys
from pathlib import Path
from app import load_corpus_embeddings, CORPUS_INDEX_FILE

def build():
    filenames, texts, embeddings = load_corpus_embeddings()
    if len(filenames) == 0:
        print("No .txt files found in the corpus/ folder. Add plain-text papers there first.")
        return
    print(f"Built index for {len(filenames)} documents. Index stored at {CORPUS_INDEX_FILE}")

if __name__ == "__main__":
    build()

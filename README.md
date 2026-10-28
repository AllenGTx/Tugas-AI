# 🎬 Cinefy - Movie Recommendation System

**Cinefy** is an intelligent, AI-powered movie recommendation web application built with Python and **Streamlit**. It delivers personalized movie recommendations using **Content-Based Filtering** algorithms with natural language feature extraction, TF-IDF vectorization, and Cosine Distance similarity.

---

## 📌 About The Application

Cinefy acts as your personal cinema movie guide, helping users discover new films that match their taste based on deep textual and metadata relationships:

- **AI Content-Based Filtering**: Recommends movies not just by single tags, but by extracting high-dimensional patterns from combined genres, plot keywords, top cast actors, and directors.
- **Smart Dual-API Synchronization**: Seamlessly pulls high-resolution posters, trailers, runtimes, taglines, and ratings from **The Movie Database (TMDB) API**, backed with an automatic **Wikipedia API fallback** for rich, comprehensive plot synopses.
- **Interactive Cinema Experience**: Features an immersive, responsive theater-themed UI with smooth staggered entrance animations, empty-state exploration of popular titles, and detailed side-by-side modal dialogs for desktop and mobile.

---

## 🧠 Machine Learning & Architecture

1. **Feature Engineering**:
   - Extraction of key metadata: `genres`, `keywords`, `top 3 actors`, and `directors`.
   - Text normalization and tokenization into a unified content vector representation.
2. **TF-IDF Vectorization**:
   - Evaluates the significance of words across the entire 4,800+ TMDB movie dataset using `TfidfVectorizer(stop_words='english')`.
3. **Cosine Similarity**:
   - Calculates the directional cosine angle between the target movie's vector and the rest of the film catalog:
     $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
   - Returns top $N$ closest matches ranked by match percentage.

---

## 🚀 Running Locally

1. Clone or download this repository:
   ```bash
   git clone https://github.com/AllenGTx/Tugas-AI.git
   cd Tugas-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
4. Open your browser at `http://localhost:8501`.

---

## 🛠️ Built With

- **Python 3.10+**
- **Streamlit** (Web Application & Interactive UI)
- **Scikit-learn** (TF-IDF & Cosine Similarity)
- **Pandas & PyArrow** (Optimized Parquet Data Processing)
- **TMDB REST API** (Live Movie Posters & Metadata)
- **Wikipedia REST API** (In-Depth Synopsis Fallback)

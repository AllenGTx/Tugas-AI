import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for Netflix/Modern Dark Theme UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
    }
    .title-text {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #E50914, #FFA800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(229, 9, 20, 0.3);
        border-color: #E50914;
    }
    .movie-title {
        font-weight: 700;
        font-size: 1rem;
        margin-top: 10px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-meta {
        color: #f5c518;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_parquet("movies_clean.parquet")
    return df

@st.cache_resource
def compute_tfidf(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'].fillna(''))
    return tfidf, tfidf_matrix

df = load_data()
tfidf, tfidf_matrix = compute_tfidf(df)

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8"
        response = requests.get(url, timeout=3)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster+Available"

def get_recommendations(selected_title, num_recommendations=10):
    match = df[df['title'] == selected_title]
    if match.empty:
        return []
    idx = match.index[0]
    movie_vector = tfidf_matrix[idx]
    similarity_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]
    
    # Filter out the selected movie itself
    filtered_indices = [i for i in sorted_indices if i != idx][:num_recommendations]
    
    recs = []
    for i in filtered_indices:
        movie = df.iloc[i]
        recs.append({
            'movie_id': movie['movie_id'],
            'title': movie['title'],
            'vote_average': movie['vote_average'],
            'overview': movie['overview'],
            'score': round(similarity_scores[i] * 100, 1)
        })
    return recs

# Header
st.markdown('<p class="title-text">🍿 Movie Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI Content-Based Filtering menggunakan TF-IDF & Cosine Similarity</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan Rekomendasi")
selected_movie = st.sidebar.selectbox(
    "Pilih Film Favorit Anda:",
    df['title'].values,
    index=int(df[df['title'] == 'The Dark Knight Rises'].index[0]) if 'The Dark Knight Rises' in df['title'].values else 0
)

num_recs = st.sidebar.slider("Jumlah Rekomendasi:", min_value=4, max_value=20, value=10, step=2)

st.sidebar.markdown("---")
st.sidebar.info("""
**Tentang Model AI:**
- **Fitur:** Ekstraksi Genre, Keywords, Top 3 Cast, & Director.
- **Vektorisasi:** TF-IDF (Term Frequency-Inverse Document Frequency).
- **Metrik Kemiripan:** Cosine Similarity.
""")

# Selected Movie Details
col1, col2 = st.columns([1, 3])
sel_row = df[df['title'] == selected_movie].iloc[0]
with col1:
    poster_url = fetch_poster(sel_row['movie_id'])
    st.image(poster_url, use_container_width=True)
with col2:
    st.subheader(sel_row['title'])
    st.write(f"⭐ **Rating:** {sel_row['vote_average']} / 10 ({int(sel_row['vote_count'])} votes)")
    st.write(f"📅 **Rilis:** {sel_row['release_date']}")
    st.write("**Sinopsis:**")
    st.write(sel_row['overview'] if pd.notna(sel_row['overview']) else "Tidak ada ringkasan.")

st.markdown("---")
st.subheader(f"✨ Rekomendasi Serupa untuk '{selected_movie}':")

# Render Recommendations in Grid
recommendations = get_recommendations(selected_movie, num_recommendations=num_recs)

cols_per_row = 5
for row_idx in range(0, len(recommendations), cols_per_row):
    row_recs = recommendations[row_idx:row_idx + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, rec in zip(cols, row_recs):
        with col:
            rec_poster = fetch_poster(rec['movie_id'])
            st.image(rec_poster, use_container_width=True)
            st.markdown(f"**{rec['title']}**")
            st.caption(f"⭐ {rec['vote_average']} | Kemiripan: {rec['score']}%")

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(
    page_title="CinemaAI - Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling with Cinema Theme, Smooth Glassmorphism & Animations
st.markdown("""
<style>
    /* Cinema Theater Background with Deep Dark Overlay */
    .stApp {
        background: linear-gradient(rgba(10, 10, 15, 0.88), rgba(10, 10, 15, 0.94)), 
                    url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Smooth Fade-in animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Hero Banner */
    .hero-container {
        text-align: center;
        padding: 30px 10px 15px 10px;
        animation: fadeInUp 0.7s ease;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8a00 50%, #e50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #cbd5e1;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Cinema Welcome Empty State */
    .empty-state-box {
        text-align: center;
        padding: 60px 20px;
        background: rgba(18, 22, 34, 0.65);
        backdrop-filter: blur(12px);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        margin: 40px auto;
        max-width: 750px;
        animation: fadeInUp 0.8s ease;
    }

    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 15px;
        display: inline-block;
        filter: drop-shadow(0 0 15px rgba(229, 9, 20, 0.4));
    }

    /* Movie Poster Cards */
    .movie-card-container {
        background: rgba(20, 24, 38, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 8px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 12px;
    }

    .movie-card-container:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 30px rgba(229, 9, 20, 0.28);
        border-color: rgba(255, 75, 75, 0.6);
    }

    .movie-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .badge-match {
        background: rgba(229, 9, 20, 0.2);
        color: #ff6b6b;
        border: 1px solid rgba(229, 9, 20, 0.4);
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-rating {
        background: rgba(245, 197, 24, 0.15);
        color: #f5c518;
        border: 1px solid rgba(245, 197, 24, 0.3);
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Selected Movie Feature Card */
    .featured-container {
        background: rgba(18, 22, 34, 0.8);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 35px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        animation: fadeInUp 0.5s ease;
    }

    /* Buttons styling */
    div.stButton > button {
        background: linear-gradient(135deg, #e50914 0%, #b80710 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 6px 12px;
        transition: all 0.3s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #ff1e2b 0%, #d60914 100%);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
        transform: translateY(-1px);
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
    return "https://via.placeholder.com/500x750/1e293b/ffffff?text=No+Poster+Available"

def get_recommendations(selected_title, num_recommendations=10):
    match = df[df['title'] == selected_title]
    if match.empty:
        return []
    idx = match.index[0]
    movie_vector = tfidf_matrix[idx]
    similarity_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]
    
    filtered_indices = [i for i in sorted_indices if i != idx][:num_recommendations]
    
    recs = []
    for i in filtered_indices:
        movie = df.iloc[i]
        recs.append({
            'movie_id': movie['movie_id'],
            'title': movie['title'],
            'vote_average': movie['vote_average'],
            'vote_count': movie.get('vote_count', 0),
            'release_date': movie.get('release_date', '-'),
            'overview': movie['overview'],
            'genres_str': movie.get('genres_str', ''),
            'cast_str': movie.get('cast_str', ''),
            'director_str': movie.get('director_str', ''),
            'score': round(similarity_scores[i] * 100, 1)
        })
    return recs

# Modal Dialog for Movie Details (Instant Popup)
@st.dialog("🎬 Detail Informasi Film")
def show_movie_modal(movie_item):
    c1, c2 = st.columns([1, 1.8])
    with c1:
        poster = fetch_poster(movie_item['movie_id'])
        st.image(poster, use_container_width=True)
    with c2:
        st.markdown(f"### {movie_item['title']}")
        st.markdown(f"⭐ **Rating:** `{movie_item['vote_average']} / 10` ({int(movie_item.get('vote_count', 0)):,} votes)")
        st.markdown(f"📅 **Tanggal Rilis:** `{movie_item.get('release_date', '-')}`")
        if movie_item.get('genres_str'):
            st.markdown(f"🎭 **Genre:** {movie_item['genres_str']}")
        if movie_item.get('director_str'):
            st.markdown(f"🎬 **Sutradara:** {movie_item['director_str']}")
        if movie_item.get('cast_str'):
            st.markdown(f"👥 **Pemeran Utama:** {movie_item['cast_str']}")
        if 'score' in movie_item:
            st.markdown(f"🎯 **Tingkat Kemiripan:** `{movie_item['score']}% match`")
        
        st.markdown("---")
        st.markdown("**Sinopsis Cerita:**")
        st.write(movie_item['overview'] if pd.notna(movie_item['overview']) and movie_item['overview'] else "Sinopsis belum tersedia.")

# --- HERO HEADER ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 CinemaAI Movie Recommender</div>
    <div class="hero-subtitle">
        Temukan rekomendasi film terbaik yang dipersonalisasi dengan kecerdasan buatan berbasis kemiripan alur cerita, genre, sutradara, dan aktor.
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### ⚙️ Pengaturan Rekomendasi")

# Default is None so input starts empty!
movie_titles = ["-- Pilih atau Cari Judul Film --"] + list(df['title'].values)
selected_movie_raw = st.sidebar.selectbox(
    "Cari Film Favorit Anda:",
    movie_titles,
    index=0
)

num_recs = st.sidebar.slider("Jumlah Rekomendasi Ditampilkan:", min_value=5, max_value=20, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tentang Sistem AI")
st.sidebar.caption("""
- **Model:** Content-Based Filtering
- **Ekstraksi Fitur:** Genre, Keywords, Top 3 Cast, & Director
- **Algoritma:** TF-IDF Vectorizer
- **Metrik:** Cosine Distance Similarity
""")

# Check if user has chosen a movie
is_movie_selected = (selected_movie_raw != "-- Pilih atau Cari Judul Film --")

if not is_movie_selected:
    # --- EMPTY STATE (Tampilan awal saat belum input) ---
    st.markdown("""
    <div class="empty-state-box">
        <div class="empty-state-icon">🍿🎞️</div>
        <h2 style="font-weight: 700; color: #f8fafc; margin-bottom: 10px;">Layar Bioskop Masih Kosong</h2>
        <p style="color: #94a3b8; font-size: 1.05rem; line-height: 1.6; max-width: 540px; margin: 0 auto 25px auto;">
            Pilih film yang Anda sukai di menu sebelah kiri (sidebar) untuk mulai memutar mesin AI dan mendapatkan rekomendasi film serupa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Menampilkan Carousel / Cuplikan Film Populer saat belum ada input
    st.markdown("<h4 style='text-align: center; color: #cbd5e1; margin-bottom: 20px;'>🔥 Rekomendasi Film Populer Pilihan:</h4>", unsafe_allow_html=True)
    popular_movies = df.sort_values(by='vote_count', ascending=False).head(5)
    cols = st.columns(5)
    for col, (_, m_row) in zip(cols, popular_movies.iterrows()):
        with col:
            st.markdown(f"""
            <div class="movie-card-container">
                <img src="{fetch_poster(m_row['movie_id'])}" style="width: 100%; border-radius: 8px;" />
                <div class="movie-card-title">{m_row['title']}</div>
                <div style="margin-top: 4px;">
                    <span class="badge-rating">⭐ {m_row['vote_average']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Detail", key=f"pop_{m_row['movie_id']}"):
                show_movie_modal(m_row.to_dict())

else:
    # --- JIKA USER SUDAH MEMILIH FILM ---
    selected_movie = selected_movie_raw
    sel_row = df[df['title'] == selected_movie].iloc[0]
    
    st.markdown('<div class="featured-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2.8])
    with col1:
        poster_url = fetch_poster(sel_row['movie_id'])
        st.image(poster_url, use_container_width=True)
    with col2:
        st.markdown(f"<h2 style='margin-top:0; color:#ffffff;'>{sel_row['title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge-rating'>⭐ {sel_row['vote_average']} / 10 ({int(sel_row['vote_count']):,} votes)</span> &nbsp; <span style='color:#94a3b8;'>📅 Rilis: {sel_row['release_date']}</span>", unsafe_allow_html=True)
        
        if sel_row.get('genres_str'):
            st.markdown(f"<p style='margin-top: 12px; color: #cbd5e1;'><strong>🎭 Genre:</strong> {sel_row['genres_str']}</p>", unsafe_allow_html=True)
        if sel_row.get('director_str'):
            st.markdown(f"<p style='color: #cbd5e1;'><strong>🎬 Sutradara:</strong> {sel_row['director_str']}</p>", unsafe_allow_html=True)
        if sel_row.get('cast_str'):
            st.markdown(f"<p style='color: #cbd5e1;'><strong>👥 Pemeran:</strong> {sel_row['cast_str']}</p>", unsafe_allow_html=True)
            
        st.markdown("**Sinopsis:**")
        st.write(sel_row['overview'] if pd.notna(sel_row['overview']) else "Sinopsis tidak tersedia.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LIST REKOMENDASI ---
    st.markdown(f"<h3 style='color: #ffffff; margin-bottom: 20px;'>🍿 Film Serupa yang Direkomendasikan AI untuk Anda:</h3>", unsafe_allow_html=True)
    
    recommendations = get_recommendations(selected_movie, num_recommendations=num_recs)
    
    cols_per_row = 5
    for row_idx in range(0, len(recommendations), cols_per_row):
        row_recs = recommendations[row_idx:row_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, rec in zip(cols, row_recs):
            with col:
                st.markdown(f"""
                <div class="movie-card-container">
                    <img src="{fetch_poster(rec['movie_id'])}" style="width: 100%; border-radius: 8px;" />
                    <div class="movie-card-title" title="{rec['title']}">{rec['title']}</div>
                    <div style="margin-top: 6px; margin-bottom: 8px;">
                        <span class="badge-rating">⭐ {rec['vote_average']}</span>
                        <span class="badge-match">{rec['score']}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Button Popup Modal Detail
                if st.button("Lihat Detail", key=f"rec_{rec['movie_id']}"):
                    show_movie_modal(rec)

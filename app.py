import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(
    page_title="Cinefy - Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling with Cinema Theme, Cinefy Branding, Perfectly Centered Background & Polished UI
st.markdown("""
<style>
    /* Fullscreen Cinema Background - Centered & Fixed */
    .stApp {
        background: linear-gradient(rgba(10, 10, 15, 0.88), rgba(10, 10, 15, 0.94)), 
                    url("https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Cinefy Brand Header */
    .brand-container {
        text-align: center;
        padding: 30px 10px 10px 10px;
    }

    .brand-tag-top {
        font-size: 0.85rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 2px;
    }

    .brand-title {
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: 5px;
        color: #e50914;
        text-shadow: 0 0 25px rgba(229, 9, 20, 0.55), 0 4px 10px rgba(0, 0, 0, 0.8);
        margin: 0;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #f1f5f9;
        margin-top: 6px;
        margin-bottom: 6px;
    }

    .brand-desc {
        font-size: 0.95rem;
        color: #94a3b8;
        letter-spacing: 1px;
        max-width: 600px;
        margin: 0 auto 10px auto;
    }

    /* Welcome Box / Empty State */
    .cinefy-empty-box {
        text-align: center;
        padding: 55px 25px;
        background: rgba(18, 22, 34, 0.72);
        backdrop-filter: blur(14px);
        border: 1px dashed rgba(229, 9, 20, 0.35);
        border-radius: 18px;
        margin: 35px auto 45px auto;
        max-width: 720px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }

    .cinefy-filmstrip-icon {
        font-size: 3.2rem;
        margin-bottom: 15px;
        filter: drop-shadow(0 0 12px rgba(229, 9, 20, 0.6));
    }

    /* Featured Selected Movie Card */
    .featured-container {
        background: rgba(18, 22, 34, 0.82);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(229, 9, 20, 0.3);
        border-radius: 18px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 35px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5);
    }

    /* Movie Cards */
    .movie-card-container {
        background: rgba(20, 24, 38, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 8px;
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease, border-color 0.3s ease;
        margin-bottom: 10px;
    }

    .movie-card-container:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 30px rgba(229, 9, 20, 0.35);
        border-color: rgba(229, 9, 20, 0.65);
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
        background: rgba(229, 9, 20, 0.22);
        color: #ff6b6b;
        border: 1px solid rgba(229, 9, 20, 0.45);
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    .badge-rating {
        background: rgba(245, 197, 24, 0.16);
        color: #f5c518;
        border: 1px solid rgba(245, 197, 24, 0.35);
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    /* Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #e50914 0%, #b80710 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 6px 12px;
        transition: all 0.25s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #ff1e2b 0%, #d60914 100%);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.45);
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

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_full_data(movie_id, fallback_info=None):
    """
    Fetches poster, full overview, tagline, runtime, genres, director, and cast
    directly from TMDB API with comprehensive fallback to local data.
    """
    data_out = {
        'poster_url': "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80",
        'tagline': "",
        'runtime': "",
        'genres': "",
        'director': "",
        'cast': "",
        'overview': ""
    }
    
    # Pre-populate with fallback info from dataframe if available
    if fallback_info:
        data_out['overview'] = fallback_info.get('overview', '') or ''
        data_out['genres'] = fallback_info.get('genres_str', '') or ''
        data_out['director'] = fallback_info.get('director_str', '') or ''
        data_out['cast'] = fallback_info.get('cast_str', '') or ''

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            d = res.json()
            poster_path = d.get('poster_path')
            if poster_path:
                data_out['poster_url'] = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            
            if d.get('tagline'):
                data_out['tagline'] = d.get('tagline')
                
            if d.get('runtime'):
                data_out['runtime'] = f"{d.get('runtime')} Menit"
                
            if d.get('overview') and len(d.get('overview').strip()) > len(data_out['overview'].strip()):
                data_out['overview'] = d.get('overview')
                
            api_genres = [g['name'] for g in d.get('genres', [])]
            if api_genres:
                data_out['genres'] = ", ".join(api_genres)
                
            credits = d.get('credits', {})
            api_cast = [c['name'] for c in credits.get('cast', [])[:5]]
            if api_cast:
                data_out['cast'] = ", ".join(api_cast)
                
            api_crew = [c['name'] for c in credits.get('crew', []) if c.get('job') == 'Director']
            if api_crew:
                data_out['director'] = ", ".join(api_crew)
    except Exception:
        pass
        
    return data_out

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
    full_info = fetch_movie_full_data(movie_item['movie_id'], fallback_info=movie_item)
    c1, c2 = st.columns([1, 1.8])
    with c1:
        st.image(full_info['poster_url'], use_container_width=True)
    with c2:
        st.markdown(f"### {movie_item['title']}")
        if full_info.get('tagline'):
            st.markdown(f"*{full_info['tagline']}*")
            
        st.markdown(f"⭐ **Rating:** `{movie_item['vote_average']} / 10` ({int(movie_item.get('vote_count', 0)):,} votes)")
        st.markdown(f"📅 **Tanggal Rilis:** `{movie_item.get('release_date', '-')}`")
        if full_info.get('runtime'):
            st.markdown(f"⏱️ **Durasi:** `{full_info['runtime']}`")
        if full_info.get('genres'):
            st.markdown(f"🎭 **Genre:** {full_info['genres']}")
        if full_info.get('director'):
            st.markdown(f"🎬 **Sutradara:** {full_info['director']}")
        if full_info.get('cast'):
            st.markdown(f"👥 **Pemeran Utama:** {full_info['cast']}")
        if 'score' in movie_item:
            st.markdown(f"🎯 **Tingkat Kemiripan:** `{movie_item['score']}% match`")
        
        st.markdown("---")
        st.markdown("**Sinopsis Lengkap:**")
        overview_text = full_info['overview'] if full_info['overview'] else movie_item.get('overview', '')
        st.write(overview_text if overview_text else "Sinopsis belum tersedia.")

# --- CINEFY BRAND HEADER ---
st.markdown("""
<div class="brand-container">
    <div class="brand-tag-top">Movie Recommendation by</div>
    <div class="brand-title">CINEFY</div>
    <div class="brand-subtitle">Your Personal Movie Guide!</div>
    <div class="brand-desc">AI-Powered Recommendations with Content-Based Similarity</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### 🎬 Cinefy Navigator")

movie_titles = ["-- Pilih atau Cari Judul Film --"] + list(df['title'].values)
selected_movie_raw = st.sidebar.selectbox(
    "Cari Film Favorit Anda:",
    movie_titles,
    index=0
)

num_recs = st.sidebar.slider("Jumlah Rekomendasi Ditampilkan:", min_value=5, max_value=20, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tentang Sistem AI Cinefy")
st.sidebar.caption("""
- **Model:** Content-Based Filtering
- **Ekstraksi Fitur:** Genre, Keywords, Top Cast, & Director
- **Algoritma:** TF-IDF Vectorizer
- **Metrik:** Cosine Distance Similarity
- **Data Fallback:** TMDB Real-time Sync
""")

is_movie_selected = (selected_movie_raw != "-- Pilih atau Cari Judul Film --")

if not is_movie_selected:
    # --- EMPTY STATE (Awal saat belum ada input) ---
    st.markdown("""
    <div class="cinefy-empty-box">
        <div class="cinefy-filmstrip-icon">🎞️</div>
        <h2 style="font-weight: 800; color: #f8fafc; margin-bottom: 8px; letter-spacing: 1px;">LAYAR BIOSKOP MASIH KOSONG</h2>
        <p style="color: #94a3b8; font-size: 1.05rem; line-height: 1.6; max-width: 540px; margin: 0 auto;">
            Pilih atau ketik judul film favorit Anda di menu navigasi sebelah kiri untuk memutar mesin AI Cinefy dan menemukan rekomendasi film yang sejiwa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Populer showcase
    st.markdown("<h4 style='text-align: center; color: #cbd5e1; margin-bottom: 20px; letter-spacing: 1px;'>🔥 PILIHAN FILM TERPOPULER:</h4>", unsafe_allow_html=True)
    popular_movies = df.sort_values(by='vote_count', ascending=False).head(5)
    cols = st.columns(5)
    for col, (_, m_row) in zip(cols, popular_movies.iterrows()):
        with col:
            m_dict = m_row.to_dict()
            full_data = fetch_movie_full_data(m_dict['movie_id'], fallback_info=m_dict)
            st.markdown(f"""
            <div class="movie-card-container">
                <img src="{full_data['poster_url']}" style="width: 100%; border-radius: 8px;" />
                <div class="movie-card-title">{m_dict['title']}</div>
                <div style="margin-top: 4px;">
                    <span class="badge-rating">⭐ {m_dict['vote_average']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Lihat Detail", key=f"pop_{m_dict['movie_id']}"):
                show_movie_modal(m_dict)

else:
    # --- JIKA USER SUDAH MEMILIH FILM ---
    selected_movie = selected_movie_raw
    sel_row = df[df['title'] == selected_movie].iloc[0]
    sel_dict = sel_row.to_dict()
    sel_full = fetch_movie_full_data(sel_dict['movie_id'], fallback_info=sel_dict)
    
    st.markdown('<div class="featured-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2.8])
    with col1:
        st.image(sel_full['poster_url'], use_container_width=True)
    with col2:
        st.markdown(f"<h2 style='margin-top:0; color:#ffffff; font-weight:800;'>{sel_dict['title']}</h2>", unsafe_allow_html=True)
        if sel_full.get('tagline'):
            st.markdown(f"<p style='color: #94a3b8; font-style: italic; margin-top: -8px;'>\"{sel_full['tagline']}\"</p>", unsafe_allow_html=True)
            
        dur_str = f"&nbsp; • &nbsp; ⏱️ {sel_full['runtime']}" if sel_full.get('runtime') else ""
        st.markdown(f"<span class='badge-rating'>⭐ {sel_dict['vote_average']} / 10 ({int(sel_dict['vote_count']):,} votes)</span> &nbsp; <span style='color:#cbd5e1;'>📅 Rilis: {sel_dict['release_date']} {dur_str}</span>", unsafe_allow_html=True)
        
        if sel_full.get('genres'):
            st.markdown(f"<p style='margin-top: 14px; margin-bottom: 6px; color: #cbd5e1;'><strong>🎭 Genre:</strong> {sel_full['genres']}</p>", unsafe_allow_html=True)
        if sel_full.get('director'):
            st.markdown(f"<p style='margin-bottom: 6px; color: #cbd5e1;'><strong>🎬 Sutradara:</strong> {sel_full['director']}</p>", unsafe_allow_html=True)
        if sel_full.get('cast'):
            st.markdown(f"<p style='margin-bottom: 12px; color: #cbd5e1;'><strong>👥 Pemeran:</strong> {sel_full['cast']}</p>", unsafe_allow_html=True)
            
        st.markdown("**Sinopsis Lengkap:**")
        st.write(sel_full['overview'] if sel_full['overview'] else sel_dict.get('overview', 'Sinopsis tidak tersedia.'))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LIST REKOMENDASI ---
    st.markdown(f"<h3 style='color: #ffffff; margin-bottom: 20px; font-weight:700;'>🍿 Rekomendasi Film Serupa dari Cinefy:</h3>", unsafe_allow_html=True)
    
    recommendations = get_recommendations(selected_movie, num_recommendations=num_recs)
    
    cols_per_row = 5
    for row_idx in range(0, len(recommendations), cols_per_row):
        row_recs = recommendations[row_idx:row_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, rec in zip(cols, row_recs):
            with col:
                rec_full = fetch_movie_full_data(rec['movie_id'], fallback_info=rec)
                st.markdown(f"""
                <div class="movie-card-container">
                    <img src="{rec_full['poster_url']}" style="width: 100%; border-radius: 8px;" />
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

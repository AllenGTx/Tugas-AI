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

# Responsive CSS: Horizontal scroll on PC, sleek compact design on Mobile, modal optimized
st.markdown("""
<style>
    /* Fixed & Centered Cinema Background */
    .stApp {
        background: linear-gradient(rgba(8, 10, 15, 0.88), rgba(8, 10, 15, 0.94)), 
                    url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center top;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Cinefy Brand Header */
    .brand-container {
        text-align: center;
        padding: 20px 10px 8px 10px;
    }

    .brand-tag-top {
        font-size: 0.8rem;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .brand-title {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: 6px;
        color: #e50914;
        text-shadow: 0 0 25px rgba(229, 9, 20, 0.55), 0 4px 12px rgba(0, 0, 0, 0.9);
        margin: 0;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #f1f5f9;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    .brand-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        letter-spacing: 0.5px;
        max-width: 600px;
        margin: 0 auto 10px auto;
    }

    /* Welcome / Empty State Box */
    .cinefy-empty-box {
        text-align: center;
        padding: 45px 20px;
        background: rgba(18, 22, 34, 0.72);
        backdrop-filter: blur(14px);
        border: 1px dashed rgba(229, 9, 20, 0.35);
        border-radius: 16px;
        margin: 25px auto 35px auto;
        max-width: 680px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .cinefy-filmstrip-icon {
        font-size: 2.8rem;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 12px rgba(229, 9, 20, 0.6));
    }

    /* Selected Movie Featured Card */
    .featured-container {
        background: rgba(18, 22, 34, 0.84);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 16px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 30px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.6);
    }

    /* Staggered Animations */
    @keyframes slideUpFade {
        0% {
            opacity: 0;
            transform: translateY(22px) scale(0.97);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    .anim-card-1 { animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
    .anim-card-2 { animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
    .anim-card-3 { animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
    .anim-card-4 { animation: slideUpFade 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
    .anim-card-5 { animation: slideUpFade 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

    /* Movie Cards */
    .movie-card-container {
        background: rgba(20, 24, 38, 0.76);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 8px;
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease;
        margin-bottom: 8px;
    }

    .movie-card-container:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 14px 30px rgba(229, 9, 20, 0.4);
        border-color: rgba(229, 9, 20, 0.7);
    }

    .movie-card-img {
        width: 100%;
        border-radius: 8px;
        aspect-ratio: 2/3;
        object-fit: cover;
    }

    .movie-card-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 8px;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .badge-match {
        background: rgba(229, 9, 20, 0.22);
        color: #ff6b6b;
        border: 1px solid rgba(229, 9, 20, 0.45);
        padding: 2px 7px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 600;
    }

    .badge-rating {
        background: rgba(245, 197, 24, 0.16);
        color: #f5c518;
        border: 1px solid rgba(245, 197, 24, 0.35);
        padding: 2px 7px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 600;
    }

    /* Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #e50914 0%, #b80710 100%);
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 5px 10px;
        transition: all 0.25s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #ff1e2b 0%, #d60914 100%);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.45);
        transform: translateY(-1px);
    }

    /* Modal Styling */
    div[data-testid="stDialog"] div[role="dialog"] {
        background: #0f121a !important;
        border: 1px solid rgba(229, 9, 20, 0.4) !important;
        border-radius: 16px !important;
    }

    /* =========================================
       DESKTOP PC STYLES (Min-width: 992px)
       ========================================= */
    @media (min-width: 992px) {
        /* Modal on PC: Side-by-side with scrollable right pane to avoid tall window */
        .modal-body-container {
            display: flex;
            flex-direction: row;
            gap: 24px;
            max-height: 72vh;
        }
        .modal-left-col {
            flex: 0 0 280px;
        }
        .modal-right-col {
            flex: 1;
            overflow-y: auto;
            padding-right: 12px;
            max-height: 70vh;
        }
        .modal-right-col::-webkit-scrollbar {
            width: 6px;
        }
        .modal-right-col::-webkit-scrollbar-thumb {
            background: rgba(229, 9, 20, 0.5);
            border-radius: 3px;
        }
    }

    /* =========================================
       MOBILE RESPONSIVE STYLES (Max-width: 768px)
       ========================================= */
    @media (max-width: 768px) {
        .brand-title {
            font-size: 2.3rem !important;
            letter-spacing: 3px !important;
        }
        .brand-subtitle {
            font-size: 0.85rem !important;
            letter-spacing: 2px !important;
        }
        .brand-desc {
            font-size: 0.78rem !important;
            padding: 0 10px;
        }
        .featured-container {
            padding: 14px !important;
            margin-top: 10px !important;
        }
        .cinefy-empty-box {
            padding: 30px 15px !important;
            margin: 15px auto !important;
        }
        .cinefy-filmstrip-icon {
            font-size: 2.2rem !important;
        }
        .movie-card-title {
            font-size: 0.82rem !important;
        }
        div.stButton > button {
            padding: 4px 6px !important;
            font-size: 0.78rem !important;
        }
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
def fetch_detailed_synopsis_fallback(title, release_date="", default_overview=""):
    """
    Fetches rich, in-depth synopsis from Wikipedia API if overview is too brief.
    """
    year = release_date[:4] if release_date and len(str(release_date)) >= 4 else ""
    queries = [f"{title} ({year} film)", f"{title} (film)", title]
    
    for q in queries:
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro=true&explaintext=true&redirects=1&titles={q}"
            res = requests.get(url, headers={'User-Agent': 'CinefyApp/1.0'}, timeout=3)
            data = res.json()
            pages = data.get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                if pid != '-1' and 'extract' in pdata and len(pdata['extract'].strip()) > 100:
                    extract = pdata['extract'].strip()
                    if 'may refer to' not in extract and 'can refer to' not in extract:
                        return extract
        except Exception:
            pass
            
    return default_overview

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_full_data(movie_id, title="", release_date="", fallback_info=None):
    """
    Fetches poster, detailed overview, tagline, runtime, genres, director, and cast
    directly from TMDB API + Wikipedia in-depth synopsis fallback.
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
                data_out['runtime'] = f"{d.get('runtime')} Minutes"
                
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

    if len(data_out['overview'].strip()) < 220 and title:
        wiki_overview = fetch_detailed_synopsis_fallback(title, release_date, default_overview=data_out['overview'])
        if len(wiki_overview) > len(data_out['overview']):
            data_out['overview'] = wiki_overview
        
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

# Modal Dialog: Horizontal layout on PC (no vertical scrolling needed), stacked on Mobile
@st.dialog("🎬 Movie Details & Overview", width="large")
def show_movie_modal(movie_item):
    full_info = fetch_movie_full_data(
        movie_item['movie_id'], 
        title=movie_item['title'], 
        release_date=movie_item.get('release_date', ''), 
        fallback_info=movie_item
    )
    
    c1, c2 = st.columns([1, 2], gap="medium")
    with c1:
        st.image(full_info['poster_url'], use_container_width=True)
    with c2:
        st.markdown(f"### {movie_item['title']}")
        if full_info.get('tagline'):
            st.markdown(f"*{full_info['tagline']}*")
            
        st.markdown(f"⭐ **Rating:** `{movie_item['vote_average']} / 10` ({int(movie_item.get('vote_count', 0)):,} votes)")
        st.markdown(f"📅 **Release Date:** `{movie_item.get('release_date', '-')}`")
        if full_info.get('runtime'):
            st.markdown(f"⏱️ **Runtime:** `{full_info['runtime']}`")
        if full_info.get('genres'):
            st.markdown(f"🎭 **Genres:** {full_info['genres']}")
        if full_info.get('director'):
            st.markdown(f"🎬 **Director:** {full_info['director']}")
        if full_info.get('cast'):
            st.markdown(f"👥 **Top Cast:** {full_info['cast']}")
        if 'score' in movie_item:
            st.markdown(f"🎯 **Similarity Score:** `{movie_item['score']}% match`")
        
        st.markdown("---")
        st.markdown("**Comprehensive Synopsis:**")
        overview_text = full_info['overview'] if full_info['overview'] else movie_item.get('overview', '')
        st.write(overview_text if overview_text else "Synopsis not available.")

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

movie_titles = ["-- Select or Search a Movie --"] + list(df['title'].values)
selected_movie_raw = st.sidebar.selectbox(
    "Choose Your Favorite Movie:",
    movie_titles,
    index=0
)

num_recs = st.sidebar.slider("Recommendations Count:", min_value=5, max_value=20, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 About Cinefy AI")
st.sidebar.caption("""
- **Model:** Content-Based Filtering
- **Features Extracted:** Genres, Keywords, Top Cast, & Directors
- **Vectorizer:** TF-IDF
- **Metric:** Cosine Similarity
- **Data Fallback:** TMDB & Wikipedia Real-Time Sync
""")

is_movie_selected = (selected_movie_raw != "-- Select or Search a Movie --")

if not is_movie_selected:
    # --- EMPTY STATE (Initial Load) ---
    st.markdown("""
    <div class="cinefy-empty-box">
        <div class="cinefy-filmstrip-icon">🎞️</div>
        <h3 style="font-weight: 800; color: #f8fafc; margin-bottom: 8px; letter-spacing: 1px;">THE CINEMA SCREEN IS WAITING</h3>
        <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6; max-width: 520px; margin: 0 auto;">
            Please select or search your favorite movie in the sidebar navigator to power up Cinefy AI and discover matching cinematic recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Popular movies showcase with smooth staggered animation
    st.markdown("<h4 style='text-align: center; color: #cbd5e1; margin-bottom: 16px; letter-spacing: 1px;'>🔥 TOP POPULAR MOVIES TO EXPLORE:</h4>", unsafe_allow_html=True)
    popular_movies = df.sort_values(by='vote_count', ascending=False).head(5)
    cols = st.columns(5)
    for idx_col, (col, (_, m_row)) in enumerate(zip(cols, popular_movies.iterrows())):
        anim_class = f"anim-card-{idx_col + 1}"
        with col:
            m_dict = m_row.to_dict()
            full_data = fetch_movie_full_data(m_dict['movie_id'], title=m_dict['title'], release_date=m_dict.get('release_date', ''), fallback_info=m_dict)
            st.markdown(f"""
            <div class="movie-card-container {anim_class}">
                <img src="{full_data['poster_url']}" class="movie-card-img" />
                <div class="movie-card-title">{m_dict['title']}</div>
                <div style="margin-top: 4px;">
                    <span class="badge-rating">⭐ {m_dict['vote_average']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Details", key=f"pop_{m_dict['movie_id']}"):
                show_movie_modal(m_dict)

else:
    # --- MOVIE SELECTED STATE ---
    selected_movie = selected_movie_raw
    sel_row = df[df['title'] == selected_movie].iloc[0]
    sel_dict = sel_row.to_dict()
    sel_full = fetch_movie_full_data(
        sel_dict['movie_id'], 
        title=sel_dict['title'], 
        release_date=sel_dict.get('release_date', ''), 
        fallback_info=sel_dict
    )
    
    st.markdown('<div class="featured-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2.5], gap="medium")
    with col1:
        st.image(sel_full['poster_url'], use_container_width=True)
    with col2:
        st.markdown(f"<h2 style='margin-top:0; color:#ffffff; font-weight:800;'>{sel_dict['title']}</h2>", unsafe_allow_html=True)
        if sel_full.get('tagline'):
            st.markdown(f"<p style='color: #94a3b8; font-style: italic; margin-top: -8px;'>\"{sel_full['tagline']}\"</p>", unsafe_allow_html=True)
            
        dur_str = f"&nbsp; • &nbsp; ⏱️ {sel_full['runtime']}" if sel_full.get('runtime') else ""
        st.markdown(f"<span class='badge-rating'>⭐ {sel_dict['vote_average']} / 10 ({int(sel_dict['vote_count']):,} votes)</span> &nbsp; <span style='color:#cbd5e1;'>📅 Release: {sel_dict['release_date']} {dur_str}</span>", unsafe_allow_html=True)
        
        if sel_full.get('genres'):
            st.markdown(f"<p style='margin-top: 12px; margin-bottom: 5px; color: #cbd5e1;'><strong>🎭 Genres:</strong> {sel_full['genres']}</p>", unsafe_allow_html=True)
        if sel_full.get('director'):
            st.markdown(f"<p style='margin-bottom: 5px; color: #cbd5e1;'><strong>🎬 Director:</strong> {sel_full['director']}</p>", unsafe_allow_html=True)
        if sel_full.get('cast'):
            st.markdown(f"<p style='margin-bottom: 10px; color: #cbd5e1;'><strong>👥 Top Cast:</strong> {sel_full['cast']}</p>", unsafe_allow_html=True)
            
        st.markdown("**Comprehensive Synopsis:**")
        st.write(sel_full['overview'] if sel_full['overview'] else "Synopsis not available.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RECOMMENDATIONS GRID WITH STAGGERED SMOOTH ANIMATION ---
    st.markdown(f"<h4 style='color: #ffffff; margin-bottom: 16px; font-weight:700;'>🍿 Similar Movies Recommended by Cinefy:</h4>", unsafe_allow_html=True)
    
    recommendations = get_recommendations(selected_movie, num_recommendations=num_recs)
    
    cols_per_row = 5
    for row_idx in range(0, len(recommendations), cols_per_row):
        row_recs = recommendations[row_idx:row_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, (col, rec) in enumerate(zip(cols, row_recs)):
            anim_class = f"anim-card-{(col_idx % 5) + 1}"
            with col:
                rec_full = fetch_movie_full_data(
                    rec['movie_id'], 
                    title=rec['title'], 
                    release_date=rec.get('release_date', ''), 
                    fallback_info=rec
                )
                st.markdown(f"""
                <div class="movie-card-container {anim_class}">
                    <img src="{rec_full['poster_url']}" class="movie-card-img" />
                    <div class="movie-card-title" title="{rec['title']}">{rec['title']}</div>
                    <div style="margin-top: 5px; margin-bottom: 6px;">
                        <span class="badge-rating">⭐ {rec['vote_average']}</span>
                        <span class="badge-match">{rec['score']}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Button Popup Modal Detail
                if st.button("View Details", key=f"rec_{rec['movie_id']}"):
                    show_movie_modal(rec)

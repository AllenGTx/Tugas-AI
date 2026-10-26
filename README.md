# 🎬 Movie Recommendation System

Aplikasi web rekomendasi film berbasis Machine Learning menggunakan algoritma **TF-IDF Vectorizer** dan **Cosine Similarity** yang dibuat menggunakan **Streamlit**.

---

## 🚀 Cara Menjalankan di Lokal

1. Pastikan Python sudah terinstal.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka browser pada alamat `http://localhost:8501`.

---

## ☁️ Cara Deploy ke Streamlit Community Cloud (Gratis & Otomatis)

1. **Upload ke GitHub:**
   - Buat repositori baru di GitHub (misal: `movie-recommender`).
   - Push folder proyek ini ke repositori tersebut:
     ```bash
     git init
     git add app.py requirements.txt movies_clean.parquet README.md .gitignore
     git commit -m "Initial commit for deployment"
     git branch -M main
     git remote add origin https://github.com/<USERNAME>/<NAMA-REPO>.git
     git push -u origin main
     ```
2. **Deploy di Streamlit Cloud:**
   - Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan login dengan akun GitHub Anda.
   - Klik **"New app"**.
   - Pilih repository Anda, branch `main`, dan set **Main file path** ke `app.py`.
   - Klik **"Deploy!"**.
   - Aplikasi akan otomatis online dalam hitungan detik. Setiap kali Anda melakukan git push, aplikasi akan ter-update otomatis.

---

## ⚡ Catatan untuk Deploy Vercel
Vercel didesain untuk Next.js/Node.js/serverless static sites dengan limit ukuran lambda & eksekusi 10-15 detik, sehingga **Streamlit Community Cloud** adalah pilihan standar industri paling optimal, otomatis, gratis, dan native untuk aplikasi Python AI/ML seperti ini.

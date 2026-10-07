import re
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import trafilatura
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec

st.set_page_config(page_title="Klasifikasi Berita", page_icon="📰")


@st.cache_resource
def muat_semua():
    w2v = Word2Vec.load("skipgram.model")
    nb = joblib.load("naive_bayes.joblib")
    stop = set(StopWordRemoverFactory().get_stop_words())  # sama dengan notebook
    stem = StemmerFactory().create_stemmer()
    return w2v, nb, stop, stem


def preprocess(teks, stop, stem):
    teks = teks.lower()
    teks = re.sub(r"http\S+|www\S+", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    tokens = [k for k in teks.split() if k not in stop]
    return stem.stem(" ".join(tokens))


def ke_vektor(teks_bersih, w2v):
    vec = [w2v.wv[k] for k in teks_bersih.split() if k in w2v.wv]
    return np.mean(vec, axis=0) if vec else np.zeros(w2v.vector_size)


def ambil_teks_dari_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return trafilatura.extract(r.text, include_comments=False, include_tables=False)


def klasifikasi_dan_tampilkan(teks):
    bersih = preprocess(teks, stop, stem)
    if not bersih:
        st.warning("Teks tidak mengandung kata yang bisa diproses.")
        return

    token = bersih.split()
    cakupan = sum(k in w2v.wv for k in token) / len(token)

    vektor = ke_vektor(bersih, w2v).reshape(1, -1)
    pred = nb.predict(vektor)[0]
    proba = nb.predict_proba(vektor)[0]

    st.success(f"Kategori: **{pred}**")
    st.bar_chart(pd.Series(proba, index=nb.classes_))
    st.caption(f"{len(token)} kata setelah preprocessing, {cakupan:.0%} dikenal model Skip-gram.")

    if cakupan < 0.5:
        st.warning(
            "Banyak kata tidak dikenal model. Berita ini mungkin di luar topik "
            "sport/finance, sehingga hasilnya kurang bisa dipercaya."
        )
    with st.expander("Teks setelah preprocessing"):
        st.write(bersih)


w2v, nb, stop, stem = muat_semua()

st.title("📰 Klasifikasi Berita: Sport vs Finance")
st.caption("Alur: link/teks → preprocessing → vektor Skip-gram → Naive Bayes → kategori")

tab_url, tab_teks = st.tabs(["Link berita", "Tempel teks"])

with tab_url:
    url = st.text_input("Tempel link berita (situs apa saja)")
    if st.button("Klasifikasikan", key="btn_url"):
        if not url.startswith(("http://", "https://")):
            st.warning("Link harus diawali http:// atau https://")
        else:
            try:
                with st.spinner("Mengambil isi berita..."):
                    teks_url = ambil_teks_dari_url(url)
                if not teks_url:
                    st.error("Isi berita tidak berhasil diambil (paywall atau dimuat lewat JavaScript). Coba tab 'Tempel teks'.")
                else:
                    with st.expander("Cuplikan teks yang diambil"):
                        st.write(teks_url[:800] + "...")
                    klasifikasi_dan_tampilkan(teks_url)
            except requests.RequestException as e:
                st.error(f"Gagal membuka link: {e}")

with tab_teks:
    teks = st.text_area("Tempel isi berita di sini", height=250)
    if st.button("Klasifikasikan", key="btn_teks"):
        if not teks.strip():
            st.warning("Isi berita masih kosong.")
        else:
            klasifikasi_dan_tampilkan(teks)
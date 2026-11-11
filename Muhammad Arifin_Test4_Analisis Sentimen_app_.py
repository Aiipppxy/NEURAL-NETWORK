"""
Dashboard Analisis Sentimen - Topik: Kualitas Udara
Model: Neural Network (MLPClassifier) + TF-IDF
Jalankan: streamlit run app_sentimen_nn.py
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

st.set_page_config(page_title="Sentimen Kualitas Udara", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
LABELS = ["positif", "netral", "negatif"]
COLORS = {"positif": "#2e9e5b", "netral": "#7f8c8d", "negatif": "#d64545"}
CMAPS = {"positif": "Greens", "netral": "Greys", "negatif": "Reds"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

# Lexicon sederhana untuk topik kualitas udara (silakan tambah / sesuaikan)
KATA_POSITIF = {
    "aman", "lega", "selamat", "bantuan", "pulih", "bersih", "segar", "sehat",
    "baik", "membaik", "turun", "cerah", "sejuk", "nyaman", "bagus", "optimis",
    "hijau", "mendukung", "berhasil", "terkendali", "hujan", "lancar", "syukur",
}
KATA_NEGATIF = {
    "polusi", "parah", "buruk", "bahaya", "berbahaya", "kabut", "asap", "tercemar",
    "sesak", "batuk", "ispa", "sakit", "kotor", "pekat", "macet", "rusak",
    "menyesakkan", "gangguan", "naik", "tidak", "sehat", "mengkhawatirkan", "korban",
    "lelah", "mengganggu", "pengap", "debu", "kebakaran",
}
# "sehat" dan "tidak" muncul di dua daftar -> dibersihkan di bawah agar tidak ambigu
KATA_NEGATIF -= {"sehat", "tidak"}

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "juga", "karena", "ada", "saya", "kita", "kami", "akan", "sudah",
    "udah", "lagi", "nya", "aja", "sih", "dong", "deh", "ya", "yg", "dgn", "utk",
    "tdk", "gak", "ga", "nggak", "atau", "oleh", "dalam", "sangat", "banget",
    "si", "para", "kok", "pun", "jadi", "bisa", "mau", "ke", "se", "rt",
}


# ---------------------------------------------------------------------------
# FUNGSI PEMROSESAN
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Cleansing: hapus URL, mention, hashtag, angka, simbol; case folding; rapikan spasi."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URL
    text = re.sub(r"@\w+", " ", text)                  # mention
    text = re.sub(r"#\w+", " ", text)                  # hashtag
    text = re.sub(r"[^A-Za-z\s]", " ", text)           # simbol, angka, emoji
    text = text.lower()                                # case folding
    text = re.sub(r"\s+", " ", text).strip()           # spasi berlebih
    return text


def label_sentiment(text: str) -> str:
    """Pelabelan rule-based berdasarkan jumlah kata positif vs negatif."""
    tokens = text.split()
    pos = sum(t in KATA_POSITIF for t in tokens)
    neg = sum(t in KATA_NEGATIF for t in tokens)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


@st.cache_data(show_spinner=False)
def load_and_prepare(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    needed = {"created_at", "username", "full_text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")
    df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"]).copy()
    df = df.drop_duplicates(subset="full_text").reset_index(drop=True)
    df["clean_text"] = df["full_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df["sentimen"] = df["clean_text"].apply(label_sentiment)
    return df


@st.cache_resource(show_spinner=False)
def train_model(texts: tuple, labels: tuple, hidden: int, max_iter: int, test_size: float):
    X_text = pd.Series(texts)
    y = pd.Series(labels)
    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=test_size, random_state=42, stratify=strat
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = tfidf.fit_transform(X_tr)
    Xte = tfidf.transform(X_te)
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
        max_iter=max_iter, random_state=42, early_stopping=False,
    )
    mlp.fit(Xtr, y_tr)
    pred = mlp.predict(Xte)
    return tfidf, mlp, y_te.tolist(), pred.tolist()


def make_wordcloud(text: str, sentiment: str):
    words = " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)
    if not words.strip():
        return None
    return WordCloud(width=700, height=400, background_color="white",
                     colormap=CMAPS[sentiment], max_words=100).generate(words)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.title("🌫️ Analisis Sentimen Kualitas Udara")
st.caption("Neural Network (MLP) + TF-IDF • Data: Twitter/X")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    up = st.file_uploader("Upload CSV (opsional)", type="csv")
    st.caption("Jika kosong, aplikasi memakai `tweet_data.csv` di folder yang sama.")
    hidden = st.number_input("Neuron hidden layer", 10, 500, 100, step=10)
    max_iter = st.slider("Max iterasi", 100, 1000, 500, step=50)
    test_size = st.slider("Porsi data uji", 0.1, 0.4, 0.2, step=0.05)

try:
    df = load_and_prepare(up if up is not None else "tweet_data.csv")
except FileNotFoundError:
    st.error("File `tweet_data.csv` tidak ditemukan. Letakkan di folder yang sama atau upload lewat sidebar.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

if len(df) < 20:
    st.warning("Data terlalu sedikit untuk melatih model (minimal puluhan tweet).")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"]
)

# ---------------------------------------------------------------------------
# TAB 1: DATA
# ---------------------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(df))
    c2.metric("Jumlah pengguna unik", df["username"].nunique())
    c3.metric("Jumlah fitur (kolom)", df.shape[1])
    st.subheader("Cuplikan data hasil cleansing & pelabelan")
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "sentimen"]],
                 width="stretch", height=380)
    st.download_button("⬇️ Unduh data berlabel", df.to_csv(index=False).encode("utf-8"),
                       "tweet_berlabel.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 2: DISTRIBUSI
# ---------------------------------------------------------------------------
with tab2:
    counts = df["sentimen"].value_counts().reindex(LABELS).fillna(0).astype(int)
    st.subheader("Distribusi Sentimen")
    cols = st.columns(3)
    for col, lab in zip(cols, LABELS):
        col.metric(f"{EMOJI[lab]} {lab.capitalize()}", int(counts[lab]),
                   f"{counts[lab] / counts.sum():.1%}", delta_color="off")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
        ax.bar_label(bars)
        ax.set_ylabel("Jumlah tweet")
        ax.set_title("Diagram Batang")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
               colors=[COLORS[l] for l in counts.index])
        ax.set_title("Diagram Lingkaran")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 3: WORDCLOUD
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("WordCloud per Sentimen")
    cols = st.columns(3)
    for col, lab in zip(cols, LABELS):
        with col:
            st.markdown(f"**{EMOJI[lab]} {lab.capitalize()}**")
            wc = make_wordcloud(" ".join(df.loc[df["sentimen"] == lab, "clean_text"]), lab)
            if wc is None:
                st.info("Tidak ada data.")
            else:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 4: MODEL & EVALUASI
# ---------------------------------------------------------------------------
with tab4:
    with st.spinner("Melatih Neural Network..."):
        tfidf, mlp, y_true, y_pred = train_model(
            tuple(df["clean_text"]), tuple(df["sentimen"]), int(hidden), int(max_iter), float(test_size)
        )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    st.subheader("Metrik Performa (rata-rata weighted)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{acc:.4f}")
    m2.metric("Presisi", f"{prec:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1-Score", f"{f1:.4f}")

    st.caption(f"Data latih: {int(len(df) * (1 - test_size))} • Data uji: {len(y_true)} "
               f"• Arsitektur: TF-IDF → Dense({int(hidden)}, ReLU) → Softmax • Optimizer: Adam")

    present = [l for l in LABELS if l in set(y_true) | set(y_pred)]
    rep = pd.DataFrame(classification_report(y_true, y_pred, labels=present,
                                             output_dict=True, zero_division=0)).T.round(4)
    left, right = st.columns(2)
    with left:
        st.markdown("**Tabel evaluasi per kelas**")
        st.dataframe(rep, width="stretch")
        st.markdown("**Grafik metrik per kelas**")
        per = rep.loc[present, ["precision", "recall", "f1-score"]]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        per.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        st.pyplot(fig)
    with right:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=present)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present,
                    yticklabels=present, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

    if hasattr(mlp, "loss_curve_"):
        st.markdown("**Kurva loss selama pelatihan**")
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.plot(mlp.loss_curve_)
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 5: PREDIKSI
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Prediksi Sentimen Teks Baru")
    teks = st.text_area("Masukkan teks:", placeholder="contoh: udara hari ini sangat buruk, kabut asap pekat sekali")
    if st.button("Prediksi", type="primary"):
        bersih = clean_text(teks)
        if not bersih:
            st.warning("Teks kosong atau tidak mengandung huruf setelah dibersihkan.")
        else:
            vec = tfidf.transform([bersih])
            pred = mlp.predict(vec)[0]
            proba = mlp.predict_proba(vec)[0]
            st.markdown(f"**Teks bersih:** `{bersih}`")
            st.markdown(f"## {EMOJI[pred]} Sentimen: **{pred.upper()}**")
            prob_df = pd.DataFrame({"Sentimen": mlp.classes_, "Probabilitas": proba})
            prob_df["Probabilitas (%)"] = (prob_df["Probabilitas"] * 100).round(2)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(prob_df[["Sentimen", "Probabilitas (%)"]], hide_index=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                ax.barh(prob_df["Sentimen"], prob_df["Probabilitas"],
                        color=[COLORS[s] for s in prob_df["Sentimen"]])
                ax.set_xlim(0, 1)
                st.pyplot(fig)

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# KONFIGURASI DASAR
# =========================================================

st.set_page_config(
    page_title="Dashboard Segmentasi E-Wallet Mahasiswa",
    page_icon="💳",
    layout="wide"
)

DATA_DIR = Path("dashboard_data")


# =========================================================
# FUNGSI BANTUAN
# =========================================================

@st.cache_data
def load_csv(filename):
    """Membaca file CSV dari folder dashboard_data."""
    path = DATA_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Gagal membaca {filename}: {e}")
        return None


def show_missing_file(filename, note=""):
    """Menampilkan pesan jika file belum tersedia."""
    st.warning(f"File `{filename}` belum tersedia di folder `dashboard_data`.")
    if note:
        st.caption(note)


def format_percent(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "-"


def get_cluster_name_map(cluster_counts, clustered_df):
    """Mengambil nama segmen dari dataset clustered jika tersedia."""
    if clustered_df is not None and "cluster" in clustered_df.columns and "nama_segmen" in clustered_df.columns:
        mapping = (
            clustered_df
            .groupby("cluster")["nama_segmen"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            .to_dict()
        )
        return mapping

    if cluster_counts is not None and "cluster" in cluster_counts.columns and "nama_segmen" in cluster_counts.columns:
        return dict(zip(cluster_counts["cluster"], cluster_counts["nama_segmen"]))

    return {}


def add_segment_name(df, name_map):
    """Menambahkan nama segmen jika kolom cluster tersedia."""
    if df is not None and "cluster" in df.columns and name_map:
        df = df.copy()
        df["nama_segmen"] = df["cluster"].map(name_map)
    return df


# =========================================================
# LOAD DATA
# =========================================================

df_clustered = load_csv("03_dataset_clustered.csv")
kmeans_eval = load_csv("kmeans_evaluation_metrics.csv")
kmeans_final = load_csv("kmeans_final_metrics.csv")
cluster_counts = load_csv("jumlah_anggota_cluster.csv")
centroid_scaled = load_csv("centroid_scaled.csv")
profil_numerik = load_csv("profil_segmen_numerik_mean.csv")
profil_kategorik = load_csv("profil_segmen_kategorik_dominan.csv")

rf_summary = load_csv("evaluasi_random_forest_summary.csv")
confusion_matrix = load_csv("confusion_matrix.csv")
classification_report = load_csv("classification_report.csv")
feature_importance = load_csv("feature_importance_random_forest.csv")
arah_fitur = load_csv("arah_fitur_terhadap_cluster.csv")
top_features = load_csv("top_5_feature_importance.csv")
prediction_result = load_csv("hasil_prediksi_random_forest.csv")

cluster_name_map = get_cluster_name_map(cluster_counts, df_clustered)

if cluster_counts is not None:
    cluster_counts = add_segment_name(cluster_counts, cluster_name_map)

if profil_numerik is not None:
    profil_numerik = add_segment_name(profil_numerik, cluster_name_map)

if centroid_scaled is not None:
    centroid_scaled = add_segment_name(centroid_scaled, cluster_name_map)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih halaman:",
    [
        "Ringkasan",
        "Evaluasi K-Means",
        "Profil Segmen",
        "Random Forest",
        "Arah Fitur ke Cluster",
        "Data Output"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard revisi K-Means k=2 dan Random Forest Classification")


# =========================================================
# HEADER
# =========================================================

st.title("Dashboard Segmentasi Pasar E-Wallet Mahasiswa")
st.caption("K-Means Clustering k=2 dan Random Forest Classification")


# =========================================================
# HALAMAN 1: RINGKASAN
# =========================================================

if page == "Ringkasan":
    st.header("Ringkasan Penelitian")

    st.markdown(
        """
        Dashboard ini menampilkan hasil segmentasi mahasiswa pengguna e-wallet menggunakan algoritma **K-Means**.
        Jumlah cluster final ditentukan berdasarkan nilai **Silhouette Score**, sehingga pada revisi ini digunakan **2 cluster**.

        Setelah label cluster terbentuk, algoritma **Random Forest** digunakan untuk mengklasifikasikan label cluster tersebut
        dan melihat fitur yang paling berperan dalam membedakan segmen.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    total_responden = len(df_clustered) if df_clustered is not None else 0
    jumlah_cluster = df_clustered["cluster"].nunique() if df_clustered is not None and "cluster" in df_clustered.columns else 0

    silhouette_value = "-"
    if kmeans_final is not None and "Metrik" in kmeans_final.columns and "Nilai" in kmeans_final.columns:
        row = kmeans_final[kmeans_final["Metrik"].astype(str).str.contains("Silhouette", case=False, na=False)]
        if not row.empty:
            silhouette_value = row["Nilai"].iloc[0]

    accuracy_value = "-"
    if rf_summary is not None and "Metrik" in rf_summary.columns and "Nilai" in rf_summary.columns:
        row = rf_summary[rf_summary["Metrik"].astype(str).str.contains("Accuracy", case=False, na=False)]
        if not row.empty:
            accuracy_value = row["Nilai"].iloc[0]

    col1.metric("Total Responden", total_responden)
    col2.metric("Jumlah Cluster", jumlah_cluster)
    col3.metric("Silhouette Score", silhouette_value)
    col4.metric("Accuracy RF", accuracy_value)

    st.subheader("Distribusi Anggota Cluster")

    if cluster_counts is not None:
        display_cols = [col for col in ["cluster", "nama_segmen", "jumlah_anggota", "persentase"] if col in cluster_counts.columns]
        st.dataframe(cluster_counts[display_cols], use_container_width=True)

        if "jumlah_anggota" in cluster_counts.columns:
            fig = px.pie(
                cluster_counts,
                names="nama_segmen" if "nama_segmen" in cluster_counts.columns else "cluster",
                values="jumlah_anggota",
                title="Proporsi Anggota Setiap Segmen"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("jumlah_anggota_cluster.csv")


# =========================================================
# HALAMAN 2: EVALUASI K-MEANS
# =========================================================

elif page == "Evaluasi K-Means":
    st.header("Evaluasi K-Means")

    st.markdown(
        """
        Evaluasi K-Means dilakukan untuk menentukan jumlah cluster yang paling sesuai.
        Metrik utama yang digunakan pada revisi ini adalah **Silhouette Score**.
        """
    )

    st.subheader("Tabel Evaluasi Beberapa Nilai k")

    if kmeans_eval is not None:
        st.dataframe(kmeans_eval, use_container_width=True)

        if "k" in kmeans_eval.columns and "silhouette_score" in kmeans_eval.columns:
            fig = px.line(
                kmeans_eval,
                x="k",
                y="silhouette_score",
                markers=True,
                title="Silhouette Score untuk Beberapa Nilai k"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "k" in kmeans_eval.columns and "wcss_inertia" in kmeans_eval.columns:
            fig = px.line(
                kmeans_eval,
                x="k",
                y="wcss_inertia",
                markers=True,
                title="Elbow Method berdasarkan WCSS/Inertia"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("kmeans_evaluation_metrics.csv")

    st.subheader("Metrik Final K-Means")

    if kmeans_final is not None:
        st.dataframe(kmeans_final, use_container_width=True)
    else:
        show_missing_file("kmeans_final_metrics.csv")


# =========================================================
# HALAMAN 3: PROFIL SEGMEN
# =========================================================

elif page == "Profil Segmen":
    st.header("Profil Segmen")

    st.markdown(
        """
        Profil segmen digunakan untuk memahami karakteristik setiap cluster.
        Profil numerik menunjukkan rata-rata fitur pada masing-masing cluster,
        sedangkan profil kategorik menunjukkan kategori yang paling dominan.
        """
    )

    st.subheader("Profil Numerik")

    if profil_numerik is not None:
        st.dataframe(profil_numerik, use_container_width=True)

        numeric_cols = [
            col for col in profil_numerik.columns
            if col not in ["cluster", "nama_segmen"] and pd.api.types.is_numeric_dtype(profil_numerik[col])
        ]

        if numeric_cols:
            selected_feature = st.selectbox("Pilih fitur untuk dibandingkan:", numeric_cols)
            fig = px.bar(
                profil_numerik,
                x="nama_segmen" if "nama_segmen" in profil_numerik.columns else "cluster",
                y=selected_feature,
                title=f"Rata-rata {selected_feature} per Segmen",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("profil_segmen_numerik_mean.csv")

    st.subheader("Profil Kategorik Dominan")

    if profil_kategorik is not None:
        profil_kategorik = add_segment_name(profil_kategorik, cluster_name_map)
        st.dataframe(profil_kategorik, use_container_width=True)
    else:
        show_missing_file("profil_segmen_kategorik_dominan.csv")

    st.subheader("Centroid Scaled")

    if centroid_scaled is not None:
        st.dataframe(centroid_scaled, use_container_width=True)
    else:
        show_missing_file("centroid_scaled.csv")


# =========================================================
# HALAMAN 4: RANDOM FOREST
# =========================================================

elif page == "Random Forest":
    st.header("Random Forest Classification")

    st.markdown(
        """
        Random Forest digunakan untuk mengklasifikasikan label cluster hasil K-Means.
        Dengan kata lain, target model adalah label **Cluster 1** dan **Cluster 2** yang telah dibentuk sebelumnya.
        """
    )

    st.subheader("Evaluasi Model")

    if rf_summary is not None:
        st.dataframe(rf_summary, use_container_width=True)

        if "Metrik" in rf_summary.columns and "Nilai" in rf_summary.columns:
            fig = px.bar(
                rf_summary,
                x="Metrik",
                y="Nilai",
                title="Ringkasan Metrik Evaluasi Random Forest",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("evaluasi_random_forest_summary.csv")

    st.subheader("Confusion Matrix")

    if confusion_matrix is not None:
        st.dataframe(confusion_matrix, use_container_width=True)

        # Membersihkan kemungkinan kolom index yang ikut tersimpan
        cm_plot = confusion_matrix.copy()
        if cm_plot.columns[0].lower().startswith("unnamed"):
            cm_plot = cm_plot.set_index(cm_plot.columns[0])

        fig = px.imshow(
            cm_plot,
            text_auto=True,
            title="Confusion Matrix Random Forest"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("confusion_matrix.csv")

    st.subheader("Classification Report")

    if classification_report is not None:
        st.dataframe(classification_report, use_container_width=True)
    else:
        show_missing_file("classification_report.csv")

    st.subheader("Feature Importance")

    if feature_importance is not None:
        st.dataframe(feature_importance, use_container_width=True)

        if "fitur" in feature_importance.columns and "importance" in feature_importance.columns:
            fig = px.bar(
                feature_importance.sort_values("importance", ascending=True),
                x="importance",
                y="fitur",
                orientation="h",
                title="Feature Importance Random Forest"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("feature_importance_random_forest.csv")


# =========================================================
# HALAMAN 5: ARAH FITUR KE CLUSTER
# =========================================================

elif page == "Arah Fitur ke Cluster":
    st.header("Arah Kecenderungan Fitur terhadap Cluster")

    st.markdown(
        """
        Feature importance hanya menunjukkan fitur mana yang paling penting.
        Untuk mengetahui fitur tersebut cenderung mencirikan cluster mana,
        nilai importance perlu dibandingkan dengan rata-rata fitur pada setiap cluster.
        """
    )

    if arah_fitur is not None:
        st.dataframe(arah_fitur, use_container_width=True)

        if "fitur" in arah_fitur.columns and "importance" in arah_fitur.columns:
            top_n = st.slider("Jumlah fitur teratas yang ditampilkan:", 5, min(13, len(arah_fitur)), 10)
            top_ar = arah_fitur.head(top_n)

            fig = px.bar(
                top_ar.sort_values("importance", ascending=True),
                x="importance",
                y="fitur",
                color="cluster_dengan_nilai_lebih_tinggi" if "cluster_dengan_nilai_lebih_tinggi" in top_ar.columns else None,
                orientation="h",
                title="Fitur Penting dan Arah Kecenderungan Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Interpretasi Otomatis")
            for _, row in top_ar.iterrows():
                fitur = row.get("fitur", "-")
                cluster_tinggi = row.get("cluster_dengan_nilai_lebih_tinggi", "-")
                importance = row.get("importance", "-")
                st.write(
                    f"- **{fitur}** memiliki importance **{importance}** dan rata-rata lebih tinggi pada **{cluster_tinggi}**."
                )
    else:
        show_missing_file(
            "arah_fitur_terhadap_cluster.csv",
            "File ini perlu dibuat dari notebook 04 agar dashboard bisa menampilkan kecenderungan fitur ke cluster."
        )


# =========================================================
# HALAMAN 6: DATA OUTPUT
# =========================================================

elif page == "Data Output":
    st.header("Data Output")

    st.markdown(
        """
        Halaman ini digunakan untuk memeriksa file output yang sudah tersedia di folder `dashboard_data`.
        """
    )

    expected_files = [
        "03_dataset_clustered.csv",
        "kmeans_evaluation_metrics.csv",
        "kmeans_final_metrics.csv",
        "jumlah_anggota_cluster.csv",
        "centroid_scaled.csv",
        "profil_segmen_numerik_mean.csv",
        "profil_segmen_kategorik_dominan.csv",
        "evaluasi_random_forest_summary.csv",
        "confusion_matrix.csv",
        "classification_report.csv",
        "feature_importance_random_forest.csv",
        "arah_fitur_terhadap_cluster.csv",
        "hasil_prediksi_random_forest.csv"
    ]

    status_rows = []
    for filename in expected_files:
        path = DATA_DIR / filename
        status_rows.append({
            "file": filename,
            "status": "Ada" if path.exists() else "Belum ada"
        })

    status_df = pd.DataFrame(status_rows)
    st.dataframe(status_df, use_container_width=True)

    st.subheader("Preview Dataset Clustered")

    if df_clustered is not None:
        st.dataframe(df_clustered.head(50), use_container_width=True)
    else:
        show_missing_file("03_dataset_clustered.csv")
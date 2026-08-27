# Dashboard Segmentasi Pasar E-Wallet Mahasiswa

Dashboard ini dibuat untuk menampilkan hasil revisi penelitian:

**Segmentasi Pasar E-Wallet berdasarkan Pengguna Mahasiswa menggunakan Algoritma K-Means dan Random Forest**

Versi ini mengikuti alur revisi:

1. K-Means menggunakan jumlah cluster final **k = 2** berdasarkan Silhouette Score.
2. Random Forest digunakan untuk mengklasifikasikan label cluster hasil K-Means.
3. Feature importance digunakan sebagai analisis tambahan untuk mengetahui fitur pembeda utama.
4. Arah kecenderungan fitur ke cluster dilihat dari perbandingan rata-rata fitur per cluster.

## Struktur Folder

```text
streamlit_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── dashboard_data/
│   └── .gitkeep
├── scripts/
│   └── prepare_dashboard_data.py
└── .streamlit/
    └── config.toml
```

## File Data yang Dibutuhkan

Letakkan file berikut di folder `dashboard_data/`:

```text
03_dataset_clustered.csv
kmeans_evaluation_metrics.csv
kmeans_final_metrics.csv
jumlah_anggota_cluster.csv
centroid_scaled.csv
profil_segmen_numerik_mean.csv
profil_segmen_kategorik_dominan.csv
evaluasi_random_forest_summary.csv
confusion_matrix.csv
classification_report.csv
feature_importance_random_forest.csv
arah_fitur_terhadap_cluster.csv
hasil_prediksi_random_forest.csv
```

File di atas dihasilkan dari notebook 03 dan 04.

## Cara Menyiapkan Data Dashboard di Google Colab

Jalankan notebook sampai selesai:

```text
01_validitas_reliabilitas.ipynb
02_preprocessing.ipynb
03_kmeans_clustering_k2.ipynb
04_random_forest_classification_k2.ipynb
```

Setelah itu jalankan script:

```python
%run /content/drive/MyDrive/TA_revisi/streamlit_dashboard/scripts/prepare_dashboard_data.py
```

Script tersebut akan menyalin output penting dari folder `data/`, `output/03_clustering/`, dan `output/04_random_forest/` ke folder `streamlit_dashboard/dashboard_data/`.

## Cara Menjalankan Dashboard di Laptop

Masuk ke folder dashboard:

```bash
cd streamlit_dashboard
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan environment.

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan dashboard:

```bash
streamlit run app.py
```

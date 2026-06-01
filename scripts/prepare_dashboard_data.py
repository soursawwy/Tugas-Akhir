from pathlib import Path
import shutil

BASE_DIR = Path("/content/drive/MyDrive/TA_revisi")

DATA_DIR = BASE_DIR / "data"
OUTPUT_03_DIR = BASE_DIR / "output" / "03_clustering"
OUTPUT_04_DIR = BASE_DIR / "output" / "04_random_forest"
DASHBOARD_DATA_DIR = BASE_DIR / "streamlit_dashboard" / "dashboard_data"

DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_COPY = [
    # Dataset hasil clustering
    (DATA_DIR / "03_dataset_clustered.csv", DASHBOARD_DATA_DIR / "03_dataset_clustered.csv"),

    # Output K-Means
    (OUTPUT_03_DIR / "kmeans_evaluation_metrics.csv", DASHBOARD_DATA_DIR / "kmeans_evaluation_metrics.csv"),
    (OUTPUT_03_DIR / "kmeans_final_metrics.csv", DASHBOARD_DATA_DIR / "kmeans_final_metrics.csv"),
    (OUTPUT_03_DIR / "jumlah_anggota_cluster.csv", DASHBOARD_DATA_DIR / "jumlah_anggota_cluster.csv"),
    (OUTPUT_03_DIR / "centroid_scaled.csv", DASHBOARD_DATA_DIR / "centroid_scaled.csv"),
    (OUTPUT_03_DIR / "profil_segmen_numerik_mean.csv", DASHBOARD_DATA_DIR / "profil_segmen_numerik_mean.csv"),
    (OUTPUT_03_DIR / "profil_segmen_kategorik_dominan.csv", DASHBOARD_DATA_DIR / "profil_segmen_kategorik_dominan.csv"),

    # Output Random Forest
    (OUTPUT_04_DIR / "evaluasi_random_forest_summary.csv", DASHBOARD_DATA_DIR / "evaluasi_random_forest_summary.csv"),
    (OUTPUT_04_DIR / "confusion_matrix.csv", DASHBOARD_DATA_DIR / "confusion_matrix.csv"),
    (OUTPUT_04_DIR / "classification_report.csv", DASHBOARD_DATA_DIR / "classification_report.csv"),
    (OUTPUT_04_DIR / "feature_importance_random_forest.csv", DASHBOARD_DATA_DIR / "feature_importance_random_forest.csv"),
    (OUTPUT_04_DIR / "arah_fitur_terhadap_cluster.csv", DASHBOARD_DATA_DIR / "arah_fitur_terhadap_cluster.csv"),
    (OUTPUT_04_DIR / "hasil_prediksi_random_forest.csv", DASHBOARD_DATA_DIR / "hasil_prediksi_random_forest.csv"),
]

copied = []
missing = []

for source, destination in FILES_TO_COPY:
    if source.exists():
        shutil.copy2(source, destination)
        copied.append(destination.name)
    else:
        missing.append(str(source))

print("Selesai menyiapkan data dashboard.")
print(f"File berhasil disalin: {len(copied)}")

for file_name in copied:
    print(f"- {file_name}")

if missing:
    print("\nFile belum ditemukan:")
    for file_path in missing:
        print(f"- {file_path}")
else:
    print("\nSemua file berhasil ditemukan dan disalin.")
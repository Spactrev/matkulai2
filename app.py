import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/datasets/openml-datasets/main/data/credit-g/credit-g.csv"

st.set_page_config(
    page_title="Credit Scoring Trustworthy AI",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Kamus normalisasi tampilan
# Bagian ini hanya mengubah tampilan agar mudah dipahami saat presentasi.
# Nilai asli tetap dipakai oleh model, supaya pipeline machine learning aman.
# -----------------------------------------------------------------------------

FEATURE_LABELS = {
    "checking_status": "Status rekening giro",
    "duration": "Durasi kredit (bulan)",
    "credit_history": "Riwayat kredit",
    "purpose": "Tujuan kredit",
    "credit_amount": "Jumlah kredit",
    "savings_status": "Status tabungan",
    "employment": "Lama bekerja",
    "installment_commitment": "Persentase cicilan dari pendapatan",
    "personal_status": "Status pribadi",
    "other_parties": "Pihak penjamin/pendamping",
    "residence_since": "Lama tinggal di tempat saat ini (tahun)",
    "property_magnitude": "Jenis aset/jaminan",
    "age": "Usia",
    "other_payment_plans": "Rencana pembayaran lain",
    "housing": "Status tempat tinggal",
    "existing_credits": "Jumlah kredit berjalan",
    "job": "Jenis pekerjaan",
    "num_dependents": "Jumlah tanggungan",
    "own_telephone": "Kepemilikan telepon",
    "foreign_worker": "Status pekerja asing",
    "age_group": "Kelompok usia",
    "class": "Kelas risiko kredit",
    "target": "Kode target",
}

FEATURE_EXPLANATIONS = {
    "checking_status": "Kondisi saldo rekening giro calon peminjam.",
    "duration": "Lama waktu pinjaman dalam bulan.",
    "credit_history": "Riwayat pembayaran kredit sebelumnya.",
    "purpose": "Alasan atau tujuan pengajuan kredit.",
    "credit_amount": "Nominal kredit yang diajukan.",
    "savings_status": "Perkiraan jumlah tabungan calon peminjam.",
    "employment": "Lama calon peminjam bekerja.",
    "installment_commitment": "Perbandingan cicilan terhadap pendapatan.",
    "personal_status": "Status personal yang terdapat pada dataset.",
    "other_parties": "Ada atau tidaknya pihak lain seperti penjamin.",
    "residence_since": "Lama calon peminjam tinggal di alamat saat ini.",
    "property_magnitude": "Jenis aset yang dimiliki atau dapat menjadi jaminan.",
    "age": "Usia calon peminjam.",
    "other_payment_plans": "Rencana pembayaran lain di luar kredit utama.",
    "housing": "Status tempat tinggal calon peminjam.",
    "existing_credits": "Jumlah kredit lain yang sedang berjalan.",
    "job": "Kategori pekerjaan calon peminjam.",
    "num_dependents": "Jumlah orang yang menjadi tanggungan.",
    "own_telephone": "Keterangan apakah calon peminjam memiliki telepon.",
    "foreign_worker": "Keterangan apakah calon peminjam berstatus pekerja asing.",
    "age_group": "Kelompok usia yang dibuat dari median usia dataset.",
}

CATEGORY_LABELS = {
    "checking_status": {
        "<0": "Saldo negatif / kurang dari 0 DM",
        "0<=X<200": "Saldo rendah / 0–200 DM",
        ">=200": "Saldo cukup tinggi / ≥200 DM",
        "no checking": "Tidak punya rekening giro",
    },
    "credit_history": {
        "no credits/all paid": "Belum punya kredit / semua kredit sudah lunas",
        "all paid": "Semua kredit sebelumnya sudah lunas",
        "existing paid": "Riwayat kredit lancar",
        "delayed previously": "Pernah terlambat bayar",
        "critical/other existing credit": "Riwayat kredit bermasalah / kredit lain masih berjalan",
    },
    "purpose": {
        "new car": "Mobil baru",
        "used car": "Mobil bekas",
        "furniture/equipment": "Furnitur atau peralatan rumah",
        "radio/tv": "Elektronik / Radio / TV",
        "domestic appliance": "Peralatan rumah tangga",
        "repairs": "Perbaikan",
        "education": "Pendidikan",
        "vacation": "Liburan",
        "retraining": "Pelatihan ulang",
        "business": "Bisnis atau usaha",
        "other": "Lainnya",
    },
    "savings_status": {
        "<100": "Tabungan rendah / <100 DM",
        "100<=X<500": "Tabungan sedang / 100–500 DM",
        "500<=X<1000": "Tabungan cukup / 500–1000 DM",
        ">=1000": "Tabungan tinggi / ≥1000 DM",
        "no known savings": "Tidak diketahui / tidak ada data tabungan",
    },
    "employment": {
        "unemployed": "Tidak bekerja",
        "<1": "Bekerja < 1 tahun",
        "1<=X<4": "Bekerja 1–4 tahun",
        "4<=X<7": "Bekerja 4–7 tahun",
        ">=7": "Bekerja ≥ 7 tahun",
    },
    "personal_status": {
        "male div/sep": "Laki-laki, cerai/berpisah",
        "female div/dep/mar": "Perempuan, cerai/menikah/ada tanggungan",
        "male single": "Laki-laki lajang",
        "male mar/wid": "Laki-laki, menikah/duda",
        "female single": "Perempuan lajang",
    },
    "other_parties": {
        "none": "Tidak ada",
        "co applicant": "Pemohon bersama",
        "guarantor": "Penjamin",
    },
    "property_magnitude": {
        "real estate": "Rumah/tanah/properti",
        "life insurance": "Asuransi jiwa / tabungan",
        "car": "Mobil/kendaraan",
        "no known property": "Tidak ada aset yang tercatat",
    },
    "other_payment_plans": {
        "bank": "Ada pembayaran lain di bank",
        "stores": "Ada cicilan/pembayaran di toko",
        "none": "Tidak ada",
    },
    "housing": {
        "rent": "Rumah sewa",
        "own": "Rumah milik sendiri",
        "for free": "Tinggal gratis / ikut keluarga",
    },
    "job": {
        "unemp/unskilled non res": "Tidak bekerja / tidak terampil",
        "unskilled resident": "Pekerja tidak terampil",
        "skilled": "Pekerja terampil",
        "high qualif/self emp/mgmt": "Profesional / wiraswasta / manajemen",
    },
    "own_telephone": {
        "none": "Tidak punya telepon",
        "yes": "Punya telepon",
    },
    "foreign_worker": {
        "yes": "Ya",
        "no": "Tidak",
    },
    "age_group": {
        "younger": "Usia di bawah median",
        "older": "Usia median atau lebih tua",
        "unknown": "Tidak diketahui",
    },
    "class": {
        "good": "Risiko kredit baik",
        "bad": "Risiko kredit buruk",
    },
}

MODEL_LABELS = {
    "Logistic Regression": "Regresi Logistik",
    "Decision Tree": "Pohon Keputusan",
    "Random Forest": "Random Forest",
}

METRIC_LABELS = {
    "Model": "Model",
    "Accuracy": "Akurasi",
    "Precision": "Presisi",
    "Recall": "Recall",
    "F1-score": "F1-score",
    "accuracy": "Akurasi",
    "precision": "Presisi",
    "recall": "Recall",
    "f1_score": "F1-score",
    "selection_rate": "Selection rate",
}


def clean_category_value(value) -> str:
    """Membersihkan nilai kategori asli dataset agar tampilan tidak membawa tanda kutip."""
    if pd.isna(value):
        return "Tidak diketahui"

    text_value = str(value).strip()

    # Dataset OpenML kadang menyimpan kategori dengan tanda kutip sebagai bagian dari teks,
    # contoh: "'>=7'". Bagian ini hanya membersihkan tampilan, bukan mengubah input model.
    while len(text_value) >= 2 and (
        (text_value[0] == text_value[-1] == "'")
        or (text_value[0] == text_value[-1] == '"')
    ):
        text_value = text_value[1:-1].strip()

    return text_value


def label_feature(column_name: str) -> str:
    return FEATURE_LABELS.get(column_name, column_name)


def label_value(column_name: str, value) -> str:
    clean_value = clean_category_value(value)
    return CATEGORY_LABELS.get(column_name, {}).get(clean_value, clean_value)


def option_sort_key(column_name: str, value) -> tuple:
    """Mengurutkan dropdown sesuai urutan kamus agar lebih natural dibaca."""
    clean_value = clean_category_value(value)
    ordered_values = list(CATEGORY_LABELS.get(column_name, {}).keys())
    if clean_value in ordered_values:
        return (0, ordered_values.index(clean_value))
    return (1, label_value(column_name, value))


def sort_category_options(column_name: str, options: list) -> list:
    return sorted(options, key=lambda value: option_sort_key(column_name, value))


def label_model(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def make_dataframe_easy_to_read(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Membuat dataframe tampilan berbahasa Indonesia tanpa mengubah data asli."""
    display_df = dataframe.copy()

    for col in display_df.columns:
        if col in CATEGORY_LABELS:
            display_df[col] = display_df[col].astype(str).map(
                lambda value, c=col: label_value(c, value)
            )

    if "target" in display_df.columns:
        display_df["target"] = display_df["target"].map(
            {0: "0 = Risiko baik", 1: "1 = Risiko buruk"}
        ).fillna(display_df["target"])

    display_df = display_df.rename(columns={col: label_feature(col) for col in display_df.columns})
    return display_df


def make_results_easy_to_read(results_df: pd.DataFrame) -> pd.DataFrame:
    display_df = results_df.copy()
    display_df["Model"] = display_df["Model"].map(label_model)
    display_df = display_df.rename(columns=METRIC_LABELS)
    return display_df


def make_fairness_easy_to_read(fairness_df: pd.DataFrame, sensitive_col: str) -> pd.DataFrame:
    display_df = fairness_df.copy()
    display_df.index = [label_value(sensitive_col, index_value) for index_value in display_df.index]
    display_df = display_df.rename(columns=METRIC_LABELS)
    display_df.index.name = label_feature(sensitive_col)
    return display_df


def make_feature_importance_easy_to_read(feature_importance_df: pd.DataFrame) -> pd.DataFrame:
    if feature_importance_df.empty:
        return feature_importance_df

    display_df = feature_importance_df.copy()
    display_df["feature"] = display_df["feature"].map(normalize_encoded_feature_name)
    display_df = display_df.rename(columns={"feature": "Fitur", "importance": "Tingkat pengaruh"})
    return display_df


def normalize_encoded_feature_name(encoded_name: str) -> str:
    """Mengubah nama fitur hasil OneHotEncoder menjadi lebih mudah dibaca."""
    if encoded_name.startswith("num__"):
        raw_feature = encoded_name.replace("num__", "")
        return label_feature(raw_feature)

    if encoded_name.startswith("cat__"):
        raw_value = encoded_name.replace("cat__", "")
        matching_columns = sorted(CATEGORY_LABELS.keys(), key=len, reverse=True)
        for col in matching_columns:
            prefix = f"{col}_"
            if raw_value.startswith(prefix):
                category_value = raw_value[len(prefix):]
                return f"{label_feature(col)} = {label_value(col, category_value)}"
        return raw_value

    return encoded_name


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load German Credit dataset from public repository."""
    return pd.read_csv(DATA_URL)


def get_feature_names_from_column_transformer(column_transformer: ColumnTransformer) -> np.ndarray:
    feature_names = []

    for name, transformer, columns in column_transformer.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
        else:
            names = columns

        feature_names.extend(names)

    return np.array(feature_names)


@st.cache_resource(show_spinner=True)
def train_models():
    df = load_data().copy()
    target_col = "class"

    df["target"] = df[target_col].map({"good": 0, "bad": 1})

    if "age" in df.columns:
        df["age_group"] = np.where(df["age"] < df["age"].median(), "younger", "older")
    else:
        df["age_group"] = "unknown"

    sensitive_col = "personal_status" if "personal_status" in df.columns else "age_group"

    X = df.drop(columns=[target_col, "target"])
    y = df["target"]
    sensitive_features = df[sensitive_col]

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive_features,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=7,
        ),
    }

    trained_models = {}
    results = []

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        trained_models[model_name] = pipeline
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": f1_score,
            "selection_rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred_best,
        sensitive_features=sensitive_test,
    )

    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(
        y_test,
        y_pred_best,
        target_names=["Risiko Baik", "Risiko Buruk"],
        zero_division=0,
        output_dict=True,
    )

    feature_importance_df = pd.DataFrame()
    classifier = best_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        feature_names = get_feature_names_from_column_transformer(best_model.named_steps["preprocess"])
        feature_importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": classifier.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(15)
        )

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_test": sensitive_test,
        "sensitive_col": sensitive_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "results_df": results_df,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "y_pred_best": y_pred_best,
        "metric_frame": metric_frame,
        "fairness_df": metric_frame.by_group,
        "cm": cm,
        "report": report,
        "feature_importance_df": feature_importance_df,
    }


def metric_card(label: str, value: str):
    st.metric(label=label, value=value)


st.title("Implementasi Credit Scoring Berbasis Trustworthy AI")
st.caption("Dashboard interaktif untuk mempresentasikan alur credit scoring, evaluasi model, dan fairness.")

with st.spinner("Menyiapkan dataset dan melatih model..."):
    artifacts = train_models()


df = artifacts["df"]
X = artifacts["X"]
results_df = artifacts["results_df"]
results_display_df = make_results_easy_to_read(results_df)
best_model_name = artifacts["best_model_name"]
best_model = artifacts["best_model"]
fairness_df = artifacts["fairness_df"]
fairness_display_df = make_fairness_easy_to_read(fairness_df, artifacts["sensitive_col"])
metric_frame = artifacts["metric_frame"]

st.sidebar.header("Navigasi")
page = st.sidebar.radio(
    "Pilih bagian:",
    [
        "Ringkasan Proyek",
        "Dataset & Pipeline",
        "Performa Model",
        "Fairness",
        "Simulasi Prediksi",
        "Trustworthy AI & SDGs",
    ],
)


if page == "Ringkasan Proyek":
    st.header("Ringkasan Proyek")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Jumlah data", f"{df.shape[0]:,}")
    with col2:
        metric_card("Jumlah fitur", f"{X.shape[1]:,}")
    with col3:
        metric_card("Model terbaik", label_model(best_model_name))
    with col4:
        best_f1 = results_df.loc[results_df["Model"] == best_model_name, "F1-score"].iloc[0]
        metric_card("F1-score terbaik", f"{best_f1:.3f}")

    st.subheader("Tujuan")
    st.write(
        "Aplikasi ini menunjukkan alur implementasi credit scoring berbasis machine learning. "
        "Tujuannya adalah membantu memahami bagaimana data calon peminjam diproses, bagaimana model dilatih, "
        "bagaimana performa dievaluasi, dan bagaimana prinsip Trustworthy AI diterapkan."
    )

    st.subheader("Alur Pipeline")
    pipeline_df = pd.DataFrame(
        [
            ["1", "Load Dataset", "Mengambil German Credit Data sebagai data mentah."],
            ["2", "Data Understanding", "Memahami isi data, jumlah fitur, target, dan kondisi dataset."],
            ["3", "Normalisasi Tampilan", "Mengubah nama fitur dan pilihan kategori menjadi bahasa Indonesia agar mudah dipresentasikan."],
            ["4", "Preprocessing", "Fitur numerik distandarisasi, fitur kategorikal diubah dengan OneHotEncoder."],
            ["5", "Training", "Melatih Regresi Logistik, Pohon Keputusan, dan Random Forest."],
            ["6", "Evaluasi Performa", "Membandingkan akurasi, presisi, recall, dan F1-score."],
            ["7", "Evaluasi Fairness", "Mengecek apakah performa model berbeda pada kelompok sensitif."],
            ["8", "Interpretasi", "Melihat fitur yang paling berpengaruh agar model lebih transparan."],
        ],
        columns=["No", "Pipeline", "Penjelasan"],
    )
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

    st.info(
        "Normalisasi di aplikasi ini hanya untuk tampilan. Nilai asli tetap dipakai di belakang layar supaya model tetap sesuai dengan dataset dan pipeline training."
    )

elif page == "Dataset & Pipeline":
    st.header("Dataset & Pipeline")
    st.subheader("Preview Dataset yang Sudah Dinormalisasi")
    st.write(
        "Tabel di bawah ini memakai nama kolom dan isi kategori dalam bahasa Indonesia. "
        "Tujuannya agar dosen dan audiens lebih mudah mengikuti isi dataset."
    )
    st.dataframe(make_dataframe_easy_to_read(df.head(10)), use_container_width=True)

    with st.expander("Lihat nama kolom asli dataset"):
        st.write(", ".join(df.columns.tolist()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Target")
        target_counts = df["target"].value_counts().sort_index()
        target_view = pd.DataFrame(
            {
                "Kelas": ["Risiko kredit baik (0)", "Risiko kredit buruk (1)"],
                "Jumlah": target_counts.values,
                "Persentase": (target_counts / target_counts.sum() * 100).round(2).values,
            }
        )
        st.dataframe(target_view, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Jenis Fitur")
        numeric_readable = [label_feature(col) for col in artifacts["numeric_features"]]
        categorical_readable = [label_feature(col) for col in artifacts["categorical_features"]]
        st.write("**Fitur numerik:**")
        st.write(", ".join(numeric_readable))
        st.write("**Fitur kategorikal:**")
        st.write(", ".join(categorical_readable))

    st.subheader("Kamus Fitur")
    feature_dictionary_df = pd.DataFrame(
        [
            [label_feature(col), col, FEATURE_EXPLANATIONS.get(col, "-")]
            for col in artifacts["numeric_features"] + artifacts["categorical_features"]
        ],
        columns=["Nama mudah dipahami", "Nama asli di dataset", "Arti fitur"],
    )
    st.dataframe(feature_dictionary_df, use_container_width=True, hide_index=True)

    st.subheader("Contoh Normalisasi Kategori")
    contoh_normalisasi = []
    for fitur, mapping in CATEGORY_LABELS.items():
        for nilai_asli, nilai_tampilan in list(mapping.items())[:5]:
            contoh_normalisasi.append([label_feature(fitur), nilai_asli, nilai_tampilan])
    st.dataframe(
        pd.DataFrame(contoh_normalisasi, columns=["Fitur", "Nilai asli dataset", "Tampilan di aplikasi"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Tabel ini menunjukkan bahwa tampilan dibuat lebih manusiawi, sedangkan nilai asli tetap dipertahankan untuk model.")

    st.subheader("Penjelasan Pipeline Preprocessing")
    preprocessing_df = pd.DataFrame(
        [
            ["Fitur numerik", "StandardScaler", "Angka dibuat berada pada skala yang lebih seimbang agar model lebih stabil."],
            ["Fitur kategorikal", "OneHotEncoder", "Pilihan berbentuk teks diubah menjadi angka biner agar bisa dibaca model."],
            ["Penggabungan proses", "ColumnTransformer", "Semua proses preprocessing digabung agar rapi dan konsisten."],
            ["Training dan prediksi", "Pipeline", "Preprocessing dan model disatukan, sehingga data baru diproses dengan cara yang sama."],
        ],
        columns=["Bagian", "Metode", "Penjelasan presentation-friendly"],
    )
    st.dataframe(preprocessing_df, use_container_width=True, hide_index=True)

elif page == "Performa Model":
    st.header("Performa Model")
    st.subheader("Perbandingan Metrik")
    st.dataframe(results_display_df.round(4), use_container_width=True, hide_index=True)

    st.write(
        "Model terbaik dipilih berdasarkan F1-score karena metrik ini menyeimbangkan presisi dan recall. "
        "Dalam kasus credit scoring, kita tidak cukup hanya melihat akurasi."
    )

    st.subheader("Visualisasi F1-score")
    chart_df = results_df.copy()
    chart_df["Model Tampilan"] = chart_df["Model"].map(label_model)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(chart_df["Model Tampilan"], chart_df["F1-score"])
    ax.set_title("Perbandingan F1-score Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1-score")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    st.subheader("Confusion Matrix Model Terbaik")
    cm = artifacts["cm"]
    cm_df = pd.DataFrame(
        cm,
        index=["Aktual: Risiko baik", "Aktual: Risiko buruk"],
        columns=["Prediksi: Risiko baik", "Prediksi: Risiko buruk"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(artifacts["report"]).T
    report_df = report_df.rename(
        columns={
            "precision": "Presisi",
            "recall": "Recall",
            "f1-score": "F1-score",
            "support": "Jumlah data",
        },
        index={
            "accuracy": "Akurasi",
            "macro avg": "Rata-rata makro",
            "weighted avg": "Rata-rata tertimbang",
        },
    )
    st.dataframe(report_df.round(4), use_container_width=True)

    feature_importance_display = make_feature_importance_easy_to_read(artifacts["feature_importance_df"])
    if not feature_importance_display.empty:
        st.subheader("Fitur yang Paling Berpengaruh")
        st.dataframe(feature_importance_display.round(4), use_container_width=True, hide_index=True)

elif page == "Fairness":
    st.header("Evaluasi Fairness")
    st.write(f"Atribut sensitif yang digunakan: **{label_feature(artifacts['sensitive_col'])}**")

    st.subheader("Metrik per Kelompok Sensitif")
    st.dataframe(fairness_display_df.round(4), use_container_width=True)

    st.subheader("Visualisasi Metrik Fairness")
    fig, ax = plt.subplots(figsize=(10, 5))
    fairness_display_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Evaluasi Fairness berdasarkan {label_feature(artifacts['sensitive_col'])}")
    ax.set_xlabel("Kelompok sensitif")
    ax.set_ylabel("Nilai metrik")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selisih antar kelompok")
        difference_df = metric_frame.difference(method="between_groups").to_frame("Selisih")
        difference_df.index = [METRIC_LABELS.get(index, index) for index in difference_df.index]
        st.dataframe(difference_df.round(4))
    with col2:
        st.subheader("Rasio antar kelompok")
        ratio_df = metric_frame.ratio(method="between_groups").to_frame("Rasio")
        ratio_df.index = [METRIC_LABELS.get(index, index) for index in ratio_df.index]
        st.dataframe(ratio_df.round(4))

    st.info(
        "Interpretasi: jika selisih antar kelompok besar atau rasionya terlalu jauh dari 1, maka model perlu diperiksa lagi. "
        "Evaluasi ini masih sederhana dan digunakan untuk pembelajaran Trustworthy AI."
    )

elif page == "Simulasi Prediksi":
    st.header("Simulasi Prediksi Risiko Kredit")
    st.write(
        "Gunakan form berikut untuk mencoba prediksi satu calon peminjam. "
        "Semua label dan isi dropdown sudah dibuat menjadi bahasa Indonesia yang lebih mudah dibaca."
    )
    with st.expander("Cara membaca input simulasi"):
        st.markdown(
            """
            - **Fitur numerik** adalah data angka, misalnya usia, durasi kredit, dan jumlah kredit.
            - **Fitur kategorikal** adalah pilihan kondisi calon peminjam, misalnya status rekening, riwayat kredit, pekerjaan, dan tempat tinggal.
            - Teks pada dropdown sudah dinormalisasi agar mudah dipresentasikan.
            - Saat tombol prediksi ditekan, aplikasi tetap mengirim **nilai asli dataset** ke pipeline model, sehingga implementasinya tetap benar.
            """
        )

    selected_idx = st.slider("Pilih contoh data dari dataset", 0, len(X) - 1, 0)
    sample = X.iloc[selected_idx].copy()

    with st.form("prediction_form"):
        input_values = {}
        numeric_features = artifacts["numeric_features"]
        categorical_features = artifacts["categorical_features"]

        st.subheader("Fitur Numerik")
        st.caption("Bagian ini berisi data angka seperti usia, durasi kredit, dan jumlah kredit.")
        num_cols = st.columns(3)
        for i, col in enumerate(numeric_features):
            value = float(sample[col])
            min_value = float(X[col].min())
            max_value = float(X[col].max())
            step = 1.0 if np.issubdtype(X[col].dtype, np.integer) else 0.1
            input_values[col] = num_cols[i % 3].number_input(
                label_feature(col),
                min_value=min_value,
                max_value=max_value,
                value=value,
                step=step,
                help=f"Nama asli dataset: {col}. {FEATURE_EXPLANATIONS.get(col, '')}",
            )

        st.subheader("Fitur Kategorikal")
        st.caption("Bagian ini berisi pilihan kategori. Nilai yang terlihat sudah diterjemahkan ke bahasa Indonesia.")
        cat_cols = st.columns(2)
        for i, col in enumerate(categorical_features):
            options = X[col].dropna().astype(str).drop_duplicates().tolist()
            options = sort_category_options(col, options)
            current = str(sample[col])
            index = options.index(current) if current in options else 0
            input_values[col] = cat_cols[i % 2].selectbox(
                label_feature(col),
                options=options,
                index=index,
                format_func=lambda value, c=col: label_value(c, value),
                help=(
                    f"Nama asli dataset: {col}. {FEATURE_EXPLANATIONS.get(col, '')} "
                    "Pilihan di layar sudah disederhanakan, tetapi nilai asli tetap dipakai oleh model."
                ),
            )

        submitted = st.form_submit_button("Lihat Hasil Prediksi")

    if submitted:
        input_df = pd.DataFrame([input_values])
        pred = best_model.predict(input_df)[0]
        label = "Risiko Kredit Buruk" if pred == 1 else "Risiko Kredit Baik"

        st.subheader("Hasil Prediksi Model")
        if pred == 1:
            st.error(f"Prediksi model: **{label}**")
        else:
            st.success(f"Prediksi model: **{label}**")

        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(input_df)[0]
            st.write(f"Probabilitas risiko kredit baik: **{proba[0]:.2%}**")
            st.write(f"Probabilitas risiko kredit buruk: **{proba[1]:.2%}**")

        with st.expander("Lihat input yang dipakai model"):
            readable_input = make_dataframe_easy_to_read(input_df)
            st.dataframe(readable_input, use_container_width=True, hide_index=True)
            st.caption("Tabel ini memakai istilah yang sudah disederhanakan supaya lebih mudah dibaca saat presentasi.")

        st.write(
            "Hasil ini kami gunakan sebagai contoh bagaimana pipeline bekerja dari input data, preprocessing, "
            "sampai model mengeluarkan prediksi. Untuk pembahasan di kelas, bagian ini bisa dipakai untuk "
            "menunjukkan perubahan hasil ketika input calon peminjam diubah."
        )

elif page == "Trustworthy AI & SDGs":
    st.header("Trustworthy AI, Regulasi, dan SDGs")

    st.subheader("Fairness / Keadilan")
    st.write(
        "Model credit scoring harus diperiksa apakah performanya berbeda secara signifikan antar kelompok sensitif. "
        "Perbedaan besar dapat menjadi indikasi potensi bias."
    )

    st.subheader("Transparency / Transparansi")
    st.write(
        "Pipeline, metrik, dan interpretasi model perlu dijelaskan agar keputusan model tidak dianggap sebagai black box. "
        "Pada aplikasi ini, nama fitur dan kategori dinormalisasi supaya prosesnya mudah dipahami audiens."
    )

    st.subheader("Accountability / Akuntabilitas")
    st.write(
        "Setiap tahap, mulai dari sumber data, preprocessing, pemilihan model, evaluasi, hingga interpretasi perlu terdokumentasi. "
        "Tujuannya agar pengembang dapat menjelaskan dan mempertanggungjawabkan proses pembuatan model."
    )

    st.subheader("Ethics / Etika")
    st.write(
        "Credit scoring berdampak pada akses keuangan seseorang. Karena itu, sistem AI harus menghindari diskriminasi, "
        "menjaga transparansi, dan tidak dipakai sebagai satu-satunya dasar keputusan tanpa pengawasan manusia."
    )

    st.subheader("Kaitan dengan SDGs")
    sdg_df = pd.DataFrame(
        [
            ["SDG 8", "Pekerjaan Layak dan Pertumbuhan Ekonomi", "Sistem kredit yang adil dapat mendukung akses pembiayaan yang lebih inklusif."],
            ["SDG 10", "Berkurangnya Kesenjangan", "Evaluasi fairness membantu mengurangi potensi diskriminasi antar kelompok."],
        ],
        columns=["SDG", "Tema", "Kaitan"],
    )
    st.dataframe(sdg_df, use_container_width=True, hide_index=True)

    st.success(
        "Inti presentasi: AI bukan hanya harus akurat, tetapi juga harus adil, transparan, dapat dipertanggungjawabkan, dan sesuai etika."
    )

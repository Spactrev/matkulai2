import warnings

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/datasets/openml-datasets/main/data/credit-g/credit-g.csv"
RANDOM_STATE = 42

st.set_page_config(
    page_title="Simulasi Risiko Kredit",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Kamus label: yang tampil di layar dibuat bahasa Indonesia,
# sedangkan nilai asli dataset tetap dipakai untuk model.
# -----------------------------------------------------------------------------
FEATURE_LABEL = {
    "checking_status": "Status rekening giro",
    "duration": "Durasi kredit (bulan)",
    "credit_history": "Riwayat kredit",
    "purpose": "Tujuan kredit",
    "credit_amount": "Jumlah kredit",
    "savings_status": "Status tabungan",
    "employment": "Lama bekerja",
    "installment_commitment": "Beban cicilan dari pendapatan",
    "personal_status": "Status pribadi",
    "other_parties": "Penjamin/pemohon bersama",
    "residence_since": "Lama tinggal di alamat sekarang",
    "property_magnitude": "Aset/jaminan",
    "age": "Usia",
    "other_payment_plans": "Pembayaran lain",
    "housing": "Tempat tinggal",
    "existing_credits": "Jumlah kredit aktif",
    "job": "Pekerjaan",
    "num_dependents": "Jumlah tanggungan",
    "own_telephone": "Telepon",
    "foreign_worker": "Pekerja asing",
}

VALUE_LABEL = {
    "checking_status": {
        "<0": "Saldo negatif",
        "0<=X<200": "Saldo rendah",
        ">=200": "Saldo cukup tinggi",
        "no checking": "Tidak punya rekening giro",
    },
    "credit_history": {
        "no credits/all paid": "Belum punya kredit / semua lunas",
        "all paid": "Semua kredit sudah lunas",
        "existing paid": "Riwayat kredit lancar",
        "delayed previously": "Pernah terlambat bayar",
        "critical/other existing credit": "Riwayat kredit bermasalah",
    },
    "purpose": {
        "new car": "Mobil baru",
        "used car": "Mobil bekas",
        "furniture/equipment": "Furnitur/peralatan rumah",
        "radio/tv": "Elektronik / TV / radio",
        "domestic appliance": "Alat rumah tangga",
        "repairs": "Perbaikan",
        "education": "Pendidikan",
        "vacation": "Liburan",
        "retraining": "Pelatihan ulang",
        "business": "Bisnis/usaha",
        "other": "Lainnya",
    },
    "savings_status": {
        "<100": "Tabungan rendah",
        "100<=X<500": "Tabungan sedang",
        "500<=X<1000": "Tabungan cukup",
        ">=1000": "Tabungan tinggi",
        "no known savings": "Tidak ada data tabungan",
    },
    "employment": {
        "unemployed": "Tidak bekerja",
        "<1": "Bekerja kurang dari 1 tahun",
        "1<=X<4": "Bekerja 1–4 tahun",
        "4<=X<7": "Bekerja 4–7 tahun",
        ">=7": "Bekerja 7 tahun atau lebih",
    },
    "personal_status": {
        "male div/sep": "Laki-laki cerai/berpisah",
        "female div/dep/mar": "Perempuan menikah/cerai/ada tanggungan",
        "male single": "Laki-laki lajang",
        "male mar/wid": "Laki-laki menikah/duda",
        "female single": "Perempuan lajang",
    },
    "other_parties": {
        "none": "Tidak ada",
        "co applicant": "Pemohon bersama",
        "guarantor": "Ada penjamin",
    },
    "property_magnitude": {
        "real estate": "Properti/rumah/tanah",
        "life insurance": "Asuransi/tabungan",
        "car": "Kendaraan",
        "no known property": "Tidak ada aset tercatat",
    },
    "other_payment_plans": {
        "bank": "Ada pembayaran lain di bank",
        "stores": "Ada cicilan di toko",
        "none": "Tidak ada",
    },
    "housing": {
        "rent": "Sewa",
        "own": "Milik sendiri",
        "for free": "Tinggal gratis/keluarga",
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
}

LOW_RISK_EXAMPLE = {
    "checking_status": ">=200",
    "duration": 12,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 1500,
    "savings_status": ">=1000",
    "employment": ">=7",
    "installment_commitment": 2,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 4,
    "property_magnitude": "real estate",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "no",
}

HIGH_RISK_EXAMPLE = {
    "checking_status": "<0",
    "duration": 36,
    "credit_history": "delayed previously",
    "purpose": "new car",
    "credit_amount": 8000,
    "savings_status": "<100",
    "employment": "<1",
    "installment_commitment": 4,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 1,
    "property_magnitude": "no known property",
    "age": 22,
    "other_payment_plans": "bank",
    "housing": "rent",
    "existing_credits": 2,
    "job": "unemp/unskilled non res",
    "num_dependents": 2,
    "own_telephone": "none",
    "foreign_worker": "yes",
}


def clean_text(value):
    """Membersihkan tanda kutip bawaan dataset seperti '>=' agar dropdown rapi."""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in ["'", '"']:
        text = text[1:-1].strip()
    return text


def show_feature(column_name):
    return FEATURE_LABEL.get(column_name, column_name)


def show_value(column_name, value):
    value = clean_text(value)
    return VALUE_LABEL.get(column_name, {}).get(value, value)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(clean_text)
    return df


@st.cache_resource(show_spinner=False)
def train_model(df):
    X = df.drop(columns="class")
    y = df["class"].map({"good": 0, "bad": 1})

    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Akurasi": accuracy_score(y_test, y_pred),
        "Presisi": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }

    return pipeline, X.columns.tolist(), numeric_cols, categorical_cols, metrics


def category_options(df, column_name):
    return sorted(df[column_name].dropna().unique().tolist(), key=lambda x: show_value(column_name, x))


def default_index(options, default_value):
    default_value = clean_text(default_value)
    return options.index(default_value) if default_value in options else 0


def categorical_input(df, column_name, default_value):
    options = category_options(df, column_name)
    return st.selectbox(
        show_feature(column_name),
        options,
        index=default_index(options, default_value),
        format_func=lambda value: show_value(column_name, value),
    )


def numeric_input(df, column_name, default_value):
    min_value = int(df[column_name].min())
    max_value = int(df[column_name].max())
    return st.number_input(
        show_feature(column_name),
        min_value=min_value,
        max_value=max_value,
        value=int(default_value),
        step=1,
    )


# -----------------------------------------------------------------------------
# Tampilan aplikasi
# -----------------------------------------------------------------------------
st.title("Simulasi Prediksi Risiko Kredit")
st.caption("Dashboard sederhana untuk menjelaskan alur input → preprocessing → model → hasil prediksi.")

try:
    df = load_data()
    pipeline, feature_cols, numeric_cols, categorical_cols, metrics = train_model(df)
except Exception as error:
    st.error("Data atau model belum berhasil dimuat.")
    st.write(error)
    st.stop()

with st.sidebar:
    st.header("Navigasi Demo")
    st.write("Gunakan urutan ini saat presentasi:")
    st.write("1. Pilih contoh skenario")
    st.write("2. Cek input utama")
    st.write("3. Klik hasil prediksi")
    st.write("4. Jelaskan alur pipeline")

    st.divider()
    st.subheader("Performa singkat")
    for name, score in metrics.items():
        st.metric(name, f"{score:.2f}")

st.subheader("1. Pilih Skenario")
scenario = st.radio(
    "Supaya demo lebih cepat, pilih salah satu contoh berikut:",
    ["Contoh risiko rendah", "Contoh risiko tinggi", "Atur sendiri"],
    horizontal=True,
)

if scenario == "Contoh risiko tinggi":
    default_data = HIGH_RISK_EXAMPLE.copy()
else:
    default_data = LOW_RISK_EXAMPLE.copy()

st.subheader("2. Input Data Calon Peminjam")
st.write("Bagian ini adalah data yang akan masuk ke pipeline model.")

left, right = st.columns(2)
input_data = {}

with left:
    st.markdown("#### Kondisi keuangan")
    input_data["checking_status"] = categorical_input(df, "checking_status", default_data["checking_status"])
    input_data["savings_status"] = categorical_input(df, "savings_status", default_data["savings_status"])
    input_data["credit_history"] = categorical_input(df, "credit_history", default_data["credit_history"])
    input_data["duration"] = numeric_input(df, "duration", default_data["duration"])
    input_data["credit_amount"] = numeric_input(df, "credit_amount", default_data["credit_amount"])

with right:
    st.markdown("#### Kondisi pribadi & pekerjaan")
    input_data["employment"] = categorical_input(df, "employment", default_data["employment"])
    input_data["age"] = numeric_input(df, "age", default_data["age"])
    input_data["housing"] = categorical_input(df, "housing", default_data["housing"])
    input_data["property_magnitude"] = categorical_input(df, "property_magnitude", default_data["property_magnitude"])
    input_data["installment_commitment"] = numeric_input(
        df, "installment_commitment", default_data["installment_commitment"]
    )

with st.expander("Input tambahan yang dipakai model"):
    col_a, col_b = st.columns(2)
    with col_a:
        input_data["purpose"] = categorical_input(df, "purpose", default_data["purpose"])
        input_data["personal_status"] = categorical_input(df, "personal_status", default_data["personal_status"])
        input_data["other_parties"] = categorical_input(df, "other_parties", default_data["other_parties"])
        input_data["residence_since"] = numeric_input(df, "residence_since", default_data["residence_since"])
        input_data["existing_credits"] = numeric_input(df, "existing_credits", default_data["existing_credits"])
    with col_b:
        input_data["other_payment_plans"] = categorical_input(
            df, "other_payment_plans", default_data["other_payment_plans"]
        )
        input_data["job"] = categorical_input(df, "job", default_data["job"])
        input_data["num_dependents"] = numeric_input(df, "num_dependents", default_data["num_dependents"])
        input_data["own_telephone"] = categorical_input(df, "own_telephone", default_data["own_telephone"])
        input_data["foreign_worker"] = categorical_input(df, "foreign_worker", default_data["foreign_worker"])

# Susun urutan kolom agar sama seperti data training.
input_df = pd.DataFrame([{col: input_data[col] for col in feature_cols}])

st.subheader("3. Hasil Prediksi")
if st.button("Lihat Hasil Prediksi", type="primary"):
    prediction = int(pipeline.predict(input_df)[0])
    class_index_bad = list(pipeline.classes_).index(1)
    risk_score = float(pipeline.predict_proba(input_df)[0][class_index_bad])

    if prediction == 1:
        st.error("Hasil model: Risiko Kredit Tinggi")
        st.write(
            "Model membaca input ini sebagai kondisi yang perlu lebih diperhatikan, "
            "misalnya karena jumlah/durasi kredit, riwayat pembayaran, tabungan, atau stabilitas pekerjaan."
        )
    else:
        st.success("Hasil model: Risiko Kredit Rendah")
        st.write(
            "Model membaca input ini sebagai kondisi yang relatif lebih aman, "
            "misalnya karena riwayat kredit, tabungan, pekerjaan, atau aset lebih stabil."
        )

    st.metric("Skor risiko dari model", f"{risk_score:.1%}")

    with st.expander("Ringkasan input yang diprediksi"):
        readable_input = {
            show_feature(col): show_value(col, input_df.loc[0, col]) if col in categorical_cols else input_df.loc[0, col]
            for col in feature_cols
        }
        st.dataframe(pd.DataFrame([readable_input]), use_container_width=True)

st.divider()
st.subheader("Alur Pipeline yang Dijelaskan ke Dosen")
st.write(
    "Input calon peminjam masuk ke DataFrame, lalu fitur angka dinormalisasi dengan StandardScaler. "
    "Fitur kategori diubah menjadi angka dengan OneHotEncoder. Setelah itu data masuk ke model Random Forest "
    "untuk menghasilkan prediksi risiko kredit."
)

with st.expander("Contekan narasi singkat"):
    st.write(
        "Di bagian simulasi ini, kami ingin menunjukkan bahwa model tidak langsung membaca teks kategori begitu saja. "
        "Data yang dipilih pengguna diproses terlebih dahulu lewat preprocessing. Setelah format datanya sesuai, "
        "barulah model melakukan prediksi apakah calon peminjam cenderung masuk risiko rendah atau risiko tinggi."
    )

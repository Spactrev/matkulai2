# Versi Streamlit Sudah Dinormalisasi ke Bahasa Indonesia

File utama yang perlu dipakai adalah `app.py`.

Perubahan utama:

1. Semua nama fitur dibuat lebih mudah dipahami dalam bahasa Indonesia.
2. Nilai kategori seperti `0<=X<200`, `male mar/wid`, `all paid`, dan sejenisnya sudah diterjemahkan di tampilan aplikasi.
3. Tabel dataset, metrik performa, fairness, confusion matrix, dan simulasi prediksi memakai bahasa Indonesia.
4. Nilai asli dataset tetap dipakai di belakang layar, jadi model machine learning tetap aman dan tidak rusak.

## Cara update di GitHub

1. Buka repository GitHub yang dipakai untuk Streamlit.
2. Replace file lama dengan file dari folder ini, terutama:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
3. Commit changes.
4. Buka Streamlit Cloud.
5. Klik **Manage app**.
6. Klik **Reboot** atau **Rerun**.

Kalau sebelumnya error dependency, deploy ulang pakai Python 3.11.

## Catatan presentasi

Di bagian simulasi prediksi, pilihan dropdown sudah ditampilkan dalam bahasa Indonesia. Namun model tetap menerima nilai asli dataset. Jadi aplikasi lebih mudah dipresentasikan, tetapi pipeline machine learning tetap konsisten.

# Versi Streamlit Simulasi Singkat

Versi ini dibuat supaya presentasi simulasi prediksi risiko kredit lebih mudah dibawakan.

## Isi utama aplikasi

1. Pilih skenario contoh:
   - Contoh risiko rendah
   - Contoh risiko tinggi
   - Atur sendiri
2. Isi input utama calon peminjam.
3. Klik **Lihat Hasil Prediksi**.
4. Jelaskan alur pipeline:
   - input data
   - preprocessing numerik dan kategorikal
   - model Random Forest
   - hasil risiko kredit

## File yang perlu di-upload ke GitHub

- `app.py`
- `requirements.txt`
- `runtime.txt`

## Cara update di Streamlit

1. Replace file lama di GitHub dengan file dari folder ini.
2. Commit changes.
3. Buka Streamlit Cloud.
4. Klik **Manage app**.
5. Klik **Clear cache and reboot**.

## Narasi singkat saat presentasi

“Pada bagian simulasi, pengguna memilih kondisi calon peminjam. Data tersebut masuk ke pipeline preprocessing. Fitur numerik dinormalisasi, fitur kategorikal diubah dengan OneHotEncoder, lalu model Random Forest memberikan prediksi risiko kredit.”

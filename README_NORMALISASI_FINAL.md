# Streamlit Credit Scoring - Versi Normalisasi Final

Versi ini memperbaiki tampilan dropdown dan tabel agar mudah dibaca saat presentasi.

Perubahan utama:

- Nilai kategori yang sebelumnya masih muncul seperti `'>=7'`, `'male single'`, atau `'critical/other existing credit'` sudah dibersihkan dan diterjemahkan.
- Tanda kutip bawaan dataset dibersihkan hanya untuk tampilan.
- Dropdown diurutkan sesuai logika kategori, bukan sekadar urutan acak dari dataset.
- Nilai asli dataset tetap dipakai di belakang layar, sehingga pipeline machine learning tetap aman.
- Ditambahkan bagian penjelasan cara membaca input simulasi.

Cara update di GitHub/Streamlit:

1. Replace `app.py` lama dengan `app.py` dari folder ini.
2. Pastikan `requirements.txt` dan `runtime.txt` tetap ikut di repository.
3. Commit changes ke GitHub.
4. Buka Streamlit Cloud.
5. Klik **Manage app**.
6. Klik **Reboot app** atau **Clear cache and reboot**.

Catatan: gunakan Python 3.11 agar dependency seperti scipy dan scikit-learn aman terinstall.

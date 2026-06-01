# 🍽️ PolaKu - Food Dataset & AI Dashboard

Sistem analitik dataset makanan Indonesia untuk pelatihan model AI rekomendasi. Platform ini menyediakan dataset gizi makanan Indonesia yang telah dipreprocessing serta dashboard visualisasi interaktif.

## 🚀 Tentang Proyek

**PolaKu** adalah platform yang menyediakan:
- **Dataset Makanan**: 2.102 makanan Indonesia dengan data nutrisi lengkap
- **Dashboard Analitik**: Visualisasi interaktif untuk eksplorasi data gizi
- **Pipeline Preprocessing**: Script otomatis untuk pembersihan dan transformasi data

## 📂 Struktur Folder

```
PolaKu/
├── 📁 data/                          # Dataset dan file hasil processing
│   ├── dataset_makanan_siap_model.csv # Dataset final siap pakai (2.102 baris)
│   ├── dataset_gizi_merge_final.csv   # Dataset hasil penggabungan
│   ├── scaler_terpadu.pkl             # Model PowerTransformer yang sudah dilatih
│   ├── dataset_gizi_polaku.csv        # Dataset hasil scraping fatsecret
|   ├── dataset_gizi_nutrition.csv     # Dataset hasil scraping nutricheck
|   └── dataset_gizi_terupdate.csv     # Dataset hasil analisis data & data wrangling
│
├── 📁 utils/                           # Notebook Jupyter untuk processing
│   ├── ScrapingDataset.ipynb         # Scraping data dari FatSecret
│   ├── MergeData.ipynb               # Penggabungan dataset
│   ├── AnalisisDataScraping.ipynb    # Proses analisis data dan data wrangling
│   ├── Preprocessing&EDA.ipynb       # Preprocessing & EDA
│   ├── ab_testing.py                 # Analisis Uji Hipotesis Statistik A/B Testing
│   └── ab_testing_results.png        # Output visualisasi grafik A/B Testing        
|
├── 📁 dashboard/
│   └── dashboard.py                  # Streamlit dashboard
│
├── polaku_logo_icon.png              # logo untuk dashboard streamlit
├── requirements.txt                  # Dependencies Python
├── README.md                         # Dokumentasi ini
└── url.txt                           # Link Streamlit Dashboard yang dideploy
```

## 🛠️ Instalasi & Setup

### Prasyarat
- Python 3.10+
- pip

### Langkah-langkah Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/username/PolaKu.git
   cd PolaKu
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv env
   # Windows:
   env\Scripts\activate
   # Linux/Mac:
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan A/B Testing**
   ```bash
   cd utils
   python ab_testing.py
   ```

5. **Jalankan dashboard**
   ```bash
   cd dashboard
   streamlit run dashboard.py
   ```

   Dashboard akan terbuka di `http://localhost:8501`

## 📊 Dataset

### Dataset Final: `dataset_makanan_siap_model.csv`

| Kolom | Deskripsi |
|-------|-----------|
| `Nama Makanan` | Nama makanan |
| `Ukuran Porsi` | Ukuran porsi (100g) |
| `Kalori (kkal)` | Jumlah kalori per 100g |
| `Karbohidrat (g)` | Karbohidrat per 100g |
| `Lemak (g)` | Lemak per 100g |
| `Protein (g)` | Protein per 100g |
| `Kategori` | Kategori: Sayur, Buah, Snack, Lauk, Masakan |
| `is_seafood` | Flag alergen seafood (0/1) |
| `is_peanut` | Flag alergen kacang (0/1) |
| `is_dairy` | Flag alergen dairy (0/1) |
| `protein_to_calorie_ratio` | Rasio protein terhadap kalori |
| `is_low_calorie` | Label rendah kalori (≤100 kkal) |
| `is_low_fat` | Label rendah lemak (≤3g) |
| `is_low_carb` | Label rendah karbo (≤10g) |
| `is_high_carb` | Label tinggi karbo (>60%) |
| `is_high_protein` | Label tinggi protein (>20%) |
| `*_scaled` | Nilai yang telah ditransformasi (Yeo-Johnson) |

### Statistik Dataset
- **Total Data**: 2.102 makanan
- **Kategori**: Sayur, Buah, Snack, Lauk, Masakan
- **Sumber Data**: Scraping dari FatSecret.co.id dan nutricheck

## 📓 Pipeline Processing Data

### 1. Scraping Dataset
Notebook: `utils/ScrapingDataset.ipynb`

Proses scraping data makanan dari FatSecret.co.id dengan:
- Paginasi otomatis (maks 10 halaman per kata kunci)
- Normalisasi porsi ke 100g
- Penyaringan minuman dan bahan mentah

**Kata kunci scraping**: Nasi, Mie, Ayam, Sapi, Daging, Ikan, Udang, Cumi, Kerang, Tahu, Tempe, Telur, Sayur, Buah, Kacang, dll.

### 2. Penggabungan Dataset
Notebook: `utils/MergeData.ipynb`

- Menggabungkan 2 sumber dataset
- Standarisasi nama kolom
- Penghapusan duplikat

### 3. Analisis Data Scraping
Notebook: `utils/AnalisisDataScraping.ipynb`

**Langkah-langkah**:
1. Dokumentasi Gathering Data: Menjelaskan tahap awal pengambilan data dari file 'dataset_gizi_merged_final.csv' yang berisi 3.207 baris data nilai gizi makanan.
2. Dokumentasi Assessing Data: Merangkum proses pengecekan struktur data, tipe data (float64 untuk nutrisi), serta identifikasi awal masalah seperti nilai kosong (missing values) dan anomali statistik (contoh: kalori maksimal 35.000 kkal).
3. Dokumentasi Cleaning Data: Menjelaskan langkah pembersihan yang dilakukan: normalisasi teks 'Nama Makanan' menggunakan regex (lowercasing, menghapus karakter spesial), menangani nilai kosong dengan dropna, dan menghapus duplikasi data.
4. Dokumentasi Exploratory Data Analysis (EDA): Menjelaskan analisis visual yang dilakukan menggunakan Histogram (distribusi), Heatmap (korelasi antar nutrisi), dan Scatter Plot untuk memverifikasi hubungan linear antara makronutrisi dan kalori.
5. Dokumentasi Deteksi Outlier Ekstrem: Menjelaskan penggunaan Boxplot untuk membuktikan adanya kesalahan input/scraping, di mana nilai nutrisi melebihi batas fisik 100 gram per porsi.

### 4. Preprocessing & Feature Engineering
Notebook: `utils/Preprocessing&EDA.ipynb`

**Langkah-langkah**:
1. **Filtering**: Menghapus minuman, bahan mentah, dan noise
2. **Penanganan Duplikat**: Prioritas bahasa Indonesia
3. **Outlier Detection**: Filter berdasarkan domain knowledge
4. **Feature Engineering**:
   - Kategori makanan otomatis
   - Flag alergen (seafood, kacang, dairy)
   - Label diet (low calorie, high protein, dll)
5. **Transformasi**: PowerTransformer (Yeo-Johnson) untuk normalization

### 5. A/B Testing
Notebook: `utils/ab_testing.py`

**Langkah-langkah**:
1. cd utils
2. python ab_testing.py

**Penjelasan Logika:**
1. Dua Skenario Pengujian: Kode tersebut menguji performa rekomendasi biasa (Varian A) melawan rekomendasi bertenaga AI PolaKu (Varian B).
2. Two-Sample T-Test: Mengevaluasi apakah durasi sesi aktif pengguna meningkat signifikan berkat kualitas rekomendasi makanan yang lebih akurat.
3. Chi-Square Test: Mengukur tingkat adopsi (menyimpan/mengklik rekomendasi menu makanan). Menggunakan tabel kontingensi $2\times2$ untuk membuktikan peningkatan tingkat konversi secara valid.
4. Visualisasi Siap Pakai: Kode ini otomatis meng-output sebuah grafik cantik (ab_testing_results.png)

## 📈 Dashboard Analitik

Dashboard Streamlit menampilkan 4 analisis utama:

### Q1: Analisis Outlier
- Box plot & violin plot distribusi nutrisi
- Identifikasi makanan dengan nilai ekstrem
- Kategori: Mild vs Extreme outlier

### Q2: Distribusi Kategori Kalori
- Proporsi rendah/sedang/tinggi kalori
- Analisis keseimbangan data untuk training AI
- Rekomendasi: SMOTE/class weighting jika imbalance >15%

### Q3: Bulking vs Cutting Profile
- Analisis kelompok makanan berdasarkan kata kunci
- Bulking Score vs Cutting Score
- Radar chart profil nutrisi

### Q4: Breakdown Makronutrien
- Komposisi karbohidrat/lemak/protein top 20 makanan
- Visualisasi sunburst & pie chart

### Bonus: Analisis Lanjutan
- Scatter plot Protein vs Kalori
- Heatmap korelasi antar nutrisi
- Analisis alergen

## 🔧 Dependencies

```
streamlit==1.57.0
pandas==3.0.3
numpy==2.4.6
plotly==6.7.0
scikit-learn==1.8.0
scipy==1.17.1
beautifulsoup4==4.14.3
requests==2.34.2
joblib==1.5.3
```

## 📝 Catatan Penting

- Dataset dalam format CSV dengan encoding UTF-8
- Nilai gizi sudah dinormalisasi per 100g
- Model scaler (`scaler_terpadu.pkl`) dibutuhkan untuk transformasi data baru
- Beberapa makanan mungkin memiliki nilai tidak valid (outlier) yang sudah difilter

## 🚀 Deployment

Dashboard sudah di-deploy di: https://polaku-dashboard-eum93ksca2mi45nmbzpmwa.streamlit.app/

## 👥 Kontributor

- **Nazril Abi Widiasto** - Pengembang
- **Revolusi Al Ghifari** - Pengembang

## 📄 Lisensi

Proyek ini menggunakan data dari FatSecret.co.id dan NutriCheck.id untuk keperluan edukasi dan riset.

---

*Dibuat untuk keperluan Capstone Project DBS 2026*
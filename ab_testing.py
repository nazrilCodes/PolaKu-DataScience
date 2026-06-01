import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visualisasi gaya Seaborn
sns.set_theme(style="whitegrid")

# ==========================================
# 1. SIMULASI DATA INTERAKSI USER (A/B TESTING)
# ==========================================
def generate_ab_test_data(num_users=1000, seed=42):
    """
    Menghasilkan data simulasi interaksi pengguna untuk A/B testing PolaKu.
    
    Varian A (Kontrol): Rekomendasi Makanan Statis / Lama
    Varian B (Treatment): Rekomendasi Makanan PolaKu AI (Yeo-Johnson Scaled)
    """
    np.random.seed(seed)
    
    # 50/50 Split Pengguna
    half_users = num_users // 2
    
    # --- Varian A (Kontrol) ---
    # 1. Engagement Time (Menit di dashboard): Rata-rata 4.2 menit, Standar Deviasi 1.5
    engagement_a = np.random.normal(loc=4.2, scale=1.5, size=half_users)
    engagement_a = np.clip(engagement_a, 0.5, 15.0) # Batasi waktu rasional
    
    # 2. Adoption Rate (0: Tidak simpan menu, 1: Simpan/Adopsi menu diet rekomendasi)
    # Tingkat konversi 35%
    adoption_a = np.random.choice([0, 1], size=half_users, p=[0.65, 0.35])
    
    # --- Varian B (Treatment - PolaKu AI) ---
    # 1. Engagement Time: Rata-rata 5.1 menit, Standar Deviasi 1.7
    engagement_b = np.random.normal(loc=5.1, scale=1.7, size=half_users)
    engagement_b = np.clip(engagement_b, 0.5, 15.0)
    
    # 2. Adoption Rate: Tingkat konversi 48% (Kenaikan karena rekomendasi lebih akurat)
    adoption_b = np.random.choice([0, 1], size=half_users, p=[0.52, 0.48])
    
    # Gabungkan ke dalam DataFrame
    data_a = pd.DataFrame({
        'user_id': [f"user_{i:04d}" for i in range(1, half_users + 1)],
        'variant': 'A_Control',
        'engagement_time_mins': engagement_a,
        'adopted_plan': adoption_a
    })
    
    data_b = pd.DataFrame({
        'user_id': [f"user_{i:04d}" for i in range(half_users + 1, num_users + 1)],
        'variant': 'B_Treatment',
        'engagement_time_mins': engagement_b,
        'adopted_plan': adoption_b
    })
    
    df_ab = pd.concat([data_a, data_b], ignore_index=True)
    return df_ab

# ==========================================
# 2. ANALISIS METRIK ENGAGEMENT (T-TEST)
# ==========================================
def analyze_engagement(df):
    """
    Melakukan uji statistik parametrik Two-Sample Independent T-Test
    untuk membandingkan rata-rata waktu interaksi pengguna.
    
    Hipotesis:
    H0: mu_A = mu_B (Tidak ada perbedaan waktu engagement antara Varian A dan B)
    H1: mu_A != mu_B (Ada perbedaan signifikan pada waktu engagement)
    """
    print("\n" + "="*50)
    print(" ANALISIS METRIK 1: USER ENGAGEMENT TIME (MINUTES)")
    print("="*50)
    
    group_a = df[df['variant'] == 'A_Control']['engagement_time_mins']
    group_b = df[df['variant'] == 'B_Treatment']['engagement_time_mins']
    
    mean_a, std_a = group_a.mean(), group_a.std()
    mean_b, std_b = group_b.mean(), group_b.std()
    
    print(f"Varian A (Kontrol)   - Rata-rata: {mean_a:.2f} menit, Std Dev: {std_a:.2f}")
    print(f"Varian B (Treatment) - Rata-rata: {mean_b:.2f} menit, Std Dev: {std_b:.2f}")
    
    # Rumus Uji T Independen:
    # t = (\bar{X}_1 - \bar{X}_2) / (s_p * sqrt(1/n_1 + 1/n_2))
    t_stat, p_val = stats.ttest_ind(group_b, group_a, equal_var=False)
    
    print(f"\nNilai T-Statistic : {t_stat:.4f}")
    print(f"Nilai P-Value     : {p_val:.4e}")
    
    alpha = 0.05
    if p_val < alpha:
        print(f"Hasil: SIGNIFIKAN secara statistik (p-value < {alpha}). H0 DITOLAK.")
        improvement = ((mean_b - mean_a) / mean_a) * 100
        print(f"Kesimpulan: Rekomendasi PolaKu AI berhasil meningkatkan waktu engagement sebesar {improvement:.2f}%!")
    else:
        print(f"Hasil: TIDAK SIGNIFIKAN secara statistik (p-value >= {alpha}). H0 GAGAL DITOLAK.")
        print("Kesimpulan: Tidak ada perbedaan performa yang signifikan secara statistik.")
        
    return t_stat, p_val

# ==========================================
# 3. ANALISIS METRIK ADOPSI MENU (CHI-SQUARE TEST)
# ==========================================
def analyze_adoption(df):
    """
    Melakukan uji statistik non-parametrik Chi-Square Test of Independence
    untuk membandingkan tingkat adopsi/konversi menu rekomendasi.
    
    Hipotesis:
    H0: Variabel varian dan tingkat adopsi bersifat independen (tidak saling mempengaruhi)
    H1: Variabel varian mempengaruhi tingkat adopsi menu diet secara signifikan
    """
    print("\n" + "="*50)
    print(" ANALISIS METRIK 2: DIET PLAN ADOPTION RATE (CONVERSION)")
    print("="*50)
    
    # Membuat tabel kontingensi (contingency table)
    contingency_table = pd.crosstab(df['variant'], df['adopted_plan'])
    print("Tabel Kontingensi (Frekuensi Observasi):")
    print(contingency_table)
    
    # Menghitung proporsi konversi
    total_a = df[df['variant'] == 'A_Control'].shape[0]
    adopted_a = df[(df['variant'] == 'A_Control') & (df['adopted_plan'] == 1)].shape[0]
    rate_a = (adopted_a / total_a) * 100
    
    total_b = df[df['variant'] == 'B_Treatment'].shape[0]
    adopted_b = df[(df['variant'] == 'B_Treatment') & (df['adopted_plan'] == 1)].shape[0]
    rate_b = (adopted_b / total_b) * 100
    
    print(f"\nTingkat Adopsi Varian A (Kontrol)   : {rate_a:.2f}%")
    print(f"Tingkat Adopsi Varian B (Treatment) : {rate_b:.2f}%")
    
    # Rumus Uji Chi-Square:
    # chi^2 = sum((O - E)^2 / E)
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nNilai Chi-Square Statistic : {chi2:.4f}")
    print(f"Nilai P-Value             : {p_val:.4e}")
    
    alpha = 0.05
    if p_val < alpha:
        print(f"Hasil: SIGNIFIKAN secara statistik (p-value < {alpha}). H0 DITOLAK.")
        lift = rate_b - rate_a
        print(f"Kesimpulan: PolaKu AI terbukti secara ilmiah menaikkan rasio konversi penyimpanan menu diet sebesar +{lift:.2f}%!")
    else:
        print(f"Hasil: TIDAK SIGNIFIKAN secara statistik (p-value >= {alpha}). H0 GAGAL DITOLAK.")
        print("Kesimpulan: Perbedaan proporsi terjadi karena faktor kebetulan (random chance).")
        
    return chi2, p_val

# ==========================================
# 4. VISUALISASI HASIL A/B TESTING
# ==========================================
def plot_ab_results(df):
    """
    Membuat grafik visualisasi hasil uji hipotesis untuk presentasi.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribusi Engagement Time (KDE Plot)
    sns.kdeplot(data=df, x='engagement_time_mins', hue='variant', fill=True, common_norm=False, palette='Set2', alpha=0.5, linewidth=2, ax=axes[0])
    axes[0].set_title('Distribusi Waktu Kunjungan Pengguna (Engagement)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Durasi Sesi (Menit)')
    axes[0].set_ylabel('Kerapatan (Density)')
    
    # Plot 2: Perbandingan Tingkat Adopsi Menu (Bar Plot)
    conversion_data = df.groupby('variant')['adopted_plan'].mean().reset_index()
    conversion_data['adopted_plan'] = conversion_data['adopted_plan'] * 100
    
    # Memperbaiki peringatan visualisasi deprecation warning pada palette Seaborn
    bars = sns.barplot(
        data=conversion_data, 
        x='variant', 
        y='adopted_plan', 
        hue='variant', 
        palette='Set2', 
        legend=False, 
        width=0.5, 
        ax=axes[1]
    )
    axes[1].set_title('Rasio Adopsi Rekomendasi Menu Diet (%)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Varian Eksperimen')
    axes[1].set_ylabel('Adoption Rate (%)')
    axes[1].set_ylim(0, 100)
    
    # Tambahkan label teks persentase di atas batang grafik
    for bar in bars.patches:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Jarak teks di atas batang
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    # Mengatasi permasalahan FileNotFoundError dengan mengambil absolute path direktori script ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'ab_testing_results.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nGrafik visualisasi hasil A/B testing disimpan di '{output_path}'")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print(" MEMULAI PIPELINE UJI HIPOTESIS A/B TESTING: POLAKU AI")
    print("="*60)
    
    # Generate data user
    df_experiment = generate_ab_test_data()
    
    # Jalankan Uji T
    analyze_engagement(df_experiment)
    
    # Jalankan Uji Chi-Square
    analyze_adoption(df_experiment)
    
    # Gambar visualisasi
    plot_ab_results(df_experiment)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import base64
import os

# Mendapatkan path absolut direktori script ini dijalankan
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Mengarahkan path logo ke satu tingkat di atas folder dashboard
LOGO_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "polaku_logo_icon.png"))

def get_base64_image(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_image(LOGO_PATH)
if logo_b64:
    logo_html_main = f'<img src="data:image/png;base64,{logo_b64}" style="vertical-align: middle; height: 80px; margin-right: 16px; margin-bottom: 8px;">'
    logo_html_sidebar = f'<img src="data:image/png;base64,{logo_b64}" style="vertical-align: middle; height: 60px; margin-right: 12px; margin-bottom: 4px;">'
    logo_html_footer = f'<img src="data:image/png;base64,{logo_b64}" style="vertical-align: middle; height: 24px; margin-right: 8px;">'
    page_icon_val = LOGO_PATH
else:
    # Fallback ke emoji jika file gambar masih tidak ditemukan saat debugging
    logo_html_main = "🍽️ "
    logo_html_sidebar = "🍽️ "
    logo_html_footer = "🍽️ "
    page_icon_val = "🍽️"

st.set_page_config(
    page_title="PolaKu Food AI Dataset Dashboard",
    page_icon=page_icon_val,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.main { background: #0f0f1a; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important; border-right: 1px solid #2d2d5b; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #252550 100%);
    border: 1px solid #3d3d7a;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .value { font-size: 2.2rem; font-weight: 700; color: #7c83fd; margin-bottom: 4px; }
.metric-card .label { font-size: 0.8rem; color: #9999cc; text-transform: uppercase; letter-spacing: 1px; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #7c83fd22, transparent);
    border-left: 4px solid #7c83fd;
    padding: 12px 20px;
    border-radius: 0 12px 12px 0;
    margin: 24px 0 16px 0;
}
.section-header h2 { color: #e0e0ff; margin: 0; font-size: 1.3rem; font-weight: 600; }
.section-header p { color: #8888bb; margin: 4px 0 0 0; font-size: 0.85rem; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1f2d4a, #1a2640);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c0d4f0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.insight-box strong { color: #7cb9fd; }

/* Warning box */
.warning-box {
    background: linear-gradient(135deg, #3a1f1f, #2d1a1a);
    border: 1px solid #7a3030;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #f0c0c0;
    font-size: 0.9rem;
}

/* Divider */
.divider { border: none; border-top: 1px solid #2d2d5b; margin: 32px 0; }

/* Plotly chart container */
.plot-container { border-radius: 16px; overflow: hidden; }

/* Sidebar labels */
.sidebar-label { color: #7c83fd; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "primary": "#7c83fd",
    "secondary": "#fd7cb0",
    "success": "#7cfdb0",
    "warning": "#fdb97c",
    "danger": "#fd7c7c",
    "info": "#7cd4fd",
    "purple": "#c77cfd",
    "teal": "#7cfdee",
}

PALETTE = [
    "#7c83fd", "#fd7cb0", "#7cfdb0", "#fdb97c",
    "#fd7c7c", "#7cd4fd", "#c77cfd", "#7cfdee",
    "#fdec7c", "#b07cfd",
]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,35,0.6)",
    font=dict(color="#c0c0e0", family="Inter"),
    title_font=dict(color="#e0e0ff", size=15, family="Inter"),
    legend=dict(
        bgcolor="rgba(20,20,50,0.8)",
        bordercolor="#3d3d7a",
        borderwidth=1,
        font=dict(color="#c0c0e0"),
    ),
    xaxis=dict(gridcolor="#2d2d5b", zerolinecolor="#3d3d7a", color="#9999cc"),
    yaxis=dict(gridcolor="#2d2d5b", zerolinecolor="#3d3d7a", color="#9999cc"),
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset_makanan_siap_model.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

with st.sidebar:
    st.markdown(f"## {logo_html_sidebar} PolaKu Dashboard", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-label">📌 Filter Dataset</p>', unsafe_allow_html=True)

    selected_kategori = st.multiselect(
        "Kategori Makanan",
        options=df["Kategori"].unique().tolist(),
        default=df["Kategori"].unique().tolist(),
    )

    kalori_range = st.slider(
        "Rentang Kalori (kkal)",
        min_value=int(df["Kalori (kkal)"].min()),
        max_value=int(df["Kalori (kkal)"].max()),
        value=(int(df["Kalori (kkal)"].min()), int(df["Kalori (kkal)"].max())),
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-label">📊 Tentang Dataset</p>', unsafe_allow_html=True)
    st.info(
        f"**{len(df):,}** total makanan\n\n"
        f"**{df['Kategori'].nunique()}** kategori\n\n"
        f"Kalori min: **{df['Kalori (kkal)'].min()}** kkal\n\n"
        f"Kalori max: **{df['Kalori (kkal)'].max()}** kkal"
    )
    st.markdown("---")
    st.markdown("**Tujuan:** Dashboard analitik untuk dataset pelatihan model AI rekomendasi makanan.")

df_filtered = df[
    df["Kategori"].isin(selected_kategori) &
    df["Kalori (kkal)"].between(kalori_range[0], kalori_range[1])
]

st.markdown(f"""
<div style="text-align:center; padding: 32px 0 16px 0;">
    <h1 style="font-size:2.8rem; font-weight:700; color:#e0e0ff; margin-bottom:6px; display: flex; align-items: center; justify-content: center;">
        {logo_html_main} PolaKu Dataset Dashboard
    </h1>
    <p style="color:#8888bb; font-size:1.05rem;">
        Analitik Komprehensif • Dataset Makanan Indonesia • Model AI Rekomendasi
    </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
metrics = [
    (k1, f"{len(df_filtered):,}", "Total Makanan"),
    (k2, f"{df_filtered['Kalori (kkal)'].mean():.0f}", "Rata² Kalori"),
    (k3, f"{df_filtered['Protein (g)'].mean():.1f}g", "Rata² Protein"),
    (k4, f"{df_filtered['Lemak (g)'].mean():.1f}g", "Rata² Lemak"),
    (k5, f"{df_filtered['Karbohidrat (g)'].mean():.1f}g", "Rata² Karbo"),
    (k6, f"{df_filtered['Kategori'].nunique()}", "Kategori"),
]
for col, val, label in metrics:
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{val}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Q1 — OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🔍 Q1 · Analisis Outlier Ekstrem — Karbohidrat, Lemak & Kalori</h2>
    <p>Seberapa banyak rentang nilai anomali per nutrisi akibat scraping error atau memang nilai gizi luar batas normal?</p>
</div>
""", unsafe_allow_html=True)

def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    extreme_lo = Q1 - 3 * IQR
    extreme_hi = Q3 + 3 * IQR
    mild = series[(series < lo) | (series > hi)]
    extreme = series[(series < extreme_lo) | (series > extreme_hi)]
    return Q1, Q3, IQR, lo, hi, extreme_lo, extreme_hi, mild, extreme

nutrients = ["Kalori (kkal)", "Karbohidrat (g)", "Lemak (g)", "Protein (g)"]
nut_colors = [COLORS["primary"], COLORS["warning"], COLORS["danger"], COLORS["success"]]

col_a, col_b = st.columns([3, 2])

with col_a:
    fig_box = go.Figure()
    for nut, color in zip(nutrients, nut_colors):
        fig_box.add_trace(go.Box(
            y=df_filtered[nut],
            name=nut.split(" ")[0],
            marker_color=color,
            boxmean=True,
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(size=2, opacity=0.4),
            line=dict(width=2),
        ))
    fig_box.update_layout(
        title="Box Plot Distribusi & Outlier per Nutrisi",
        **PLOTLY_THEME,
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_b:
    rows = []
    for nut in nutrients:
        s = df_filtered[nut].dropna()
        Q1, Q3, IQR, lo, hi, elo, ehi, mild, extreme = iqr_outliers(s)
        rows.append({
            "Nutrisi": nut.split(" ")[0],
            "IQR": f"{IQR:.1f}",
            "Batas Normal": f"{max(0,lo):.1f} – {hi:.1f}",
            "Mild Outlier": len(mild),
            "Extreme Outlier": len(extreme),
            "% Extreme": f"{len(extreme)/len(s)*100:.1f}%",
        })
    df_out = pd.DataFrame(rows)
    st.markdown("**Ringkasan Outlier per Nutrisi**")
    
    st.dataframe(
        df_out.style
            .set_properties(**{"text-align": "center"})
            .map(lambda v: "color: #fd7c7c; font-weight:bold" if isinstance(v, int) and v > 20 else "", subset=["Extreme Outlier"])
        ,
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("""
    <div class="insight-box">
    <strong>📌 Interpretasi:</strong><br>
    Mild outlier = 1.5×IQR, Extreme = 3×IQR dari batas kuartil.<br>
    Lemak memiliki distribusi paling <em>right-skewed</em> wajar untuk makanan gorengan & kacang.
    </div>
    """, unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    fig_vio = go.Figure()
    for nut, color in zip(["Kalori (kkal)", "Lemak (g)"], [COLORS["primary"], COLORS["danger"]]):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        rgba_color = f"rgba({r}, {g}, {b}, 0.33)"
        
        fig_vio.add_trace(go.Violin(
            y=df_filtered[nut], name=nut.split(" ")[0],
            fillcolor=rgba_color, line_color=color,
            meanline_visible=True, box_visible=True,
            points="outliers",
        ))
    fig_vio.update_layout(title="Violin Plot: Kalori & Lemak", **PLOTLY_THEME, height=350)
    st.plotly_chart(fig_vio, use_container_width=True)

with col_d:
    fig_vio2 = go.Figure()
    for nut, color in zip(["Karbohidrat (g)", "Protein (g)"], [COLORS["warning"], COLORS["success"]]):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        rgba_color = f"rgba({r}, {g}, {b}, 0.33)"
        
        fig_vio2.add_trace(go.Violin(
            y=df_filtered[nut], name=nut.split(" ")[0],
            fillcolor=rgba_color, line_color=color,
            meanline_visible=True, box_visible=True,
            points="outliers",
        ))
    fig_vio2.update_layout(title="Violin Plot: Karbohidrat & Protein", **PLOTLY_THEME, height=350)
    st.plotly_chart(fig_vio2, use_container_width=True)

st.markdown("**🚨 Makanan dengan Nilai Ekstrem (Top Outlier per Nutrisi)**")
col_e, col_f, col_g = st.columns(3)
for col, nut, color in zip([col_e, col_f, col_g],
                            ["Kalori (kkal)", "Lemak (g)", "Protein (g)"],
                            [COLORS["primary"], COLORS["danger"], COLORS["success"]]):
    top5 = df_filtered.nlargest(5, nut)[["Nama Makanan", nut, "Kategori"]]
    fig_bar = px.bar(
        top5, x=nut, y="Nama Makanan", orientation="h",
        color_discrete_sequence=[color],
        title=f"Top 5 Tertinggi — {nut.split(' ')[0]}",
    )
    fig_bar.update_layout(**PLOTLY_THEME, height=280, margin=dict(l=0))
    col.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Q2 — KALORI CATEGORY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>⚖️ Q2 · Distribusi Kategori Kalori & Keseimbangan Data untuk Pelatihan AI</h2>
    <p>Rendah (&lt;100 kkal) · Sedang (100–250 kkal) · Tinggi (&gt;250 kkal) per 100g</p>
</div>
""", unsafe_allow_html=True)

df_filtered2 = df_filtered.copy()
df_filtered2["Kat Kalori"] = pd.cut(
    df_filtered2["Kalori (kkal)"],
    bins=[-np.inf, 100, 250, np.inf],
    labels=["🟢 Rendah (<100)", "🟡 Sedang (100-250)", "🔴 Tinggi (>250)"],
)
kat_counts = df_filtered2["Kat Kalori"].value_counts().sort_index()
kat_pct = kat_counts / kat_counts.sum() * 100

col1, col2, col3 = st.columns(3)
cat_colors_list = [COLORS["success"], COLORS["warning"], COLORS["danger"]]

with col1:
    fig_pie = go.Figure(go.Pie(
        labels=kat_counts.index,
        values=kat_counts.values,
        hole=0.55,
        marker=dict(colors=cat_colors_list, line=dict(color="#0f0f1a", width=3)),
        textinfo="percent+label",
        textfont=dict(color="#e0e0ff", size=11),
    ))
    fig_pie.update_layout(
        title="Proporsi 3 Kategori Kalori",
        **PLOTLY_THEME,
        height=350,
        showlegend=False,
        annotations=[dict(text=f"<b>{len(df_filtered2)}</b><br>Makanan", x=0.5, y=0.5,
                          font=dict(size=14, color="#e0e0ff"), showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar2 = px.bar(
        x=kat_counts.index, y=kat_counts.values,
        color=kat_counts.index,
        color_discrete_sequence=cat_colors_list,
        text=kat_counts.values,
        title="Jumlah Makanan per Kategori Kalori",
    )
    fig_bar2.update_traces(textposition="outside", textfont=dict(color="#e0e0ff"))
    fig_bar2.update_layout(**PLOTLY_THEME, height=350, showlegend=False,
                            xaxis_title="", yaxis_title="Jumlah Makanan")
    st.plotly_chart(fig_bar2, use_container_width=True)

with col3:
    kat_cross = df_filtered2.groupby(["Kategori", "Kat Kalori"]).size().reset_index(name="n")
    fig_cross = px.bar(
        kat_cross, x="Kategori", y="n", color="Kat Kalori",
        color_discrete_sequence=cat_colors_list,
        title="Distribusi Kalori per Kategori Makanan",
        barmode="stack",
    )
    fig_cross.update_layout(**PLOTLY_THEME, height=350, xaxis_title="", yaxis_title="Jumlah")
    st.plotly_chart(fig_cross, use_container_width=True)

fig_hist = px.histogram(
    df_filtered2, x="Kalori (kkal)", nbins=60,
    color="Kat Kalori",
    color_discrete_sequence=cat_colors_list,
    title="Histogram Distribusi Kalori (seluruh dataset)",
    marginal="rug",
    barmode="overlay",
)
fig_hist.update_traces(opacity=0.75)
fig_hist.update_layout(**PLOTLY_THEME, height=360)
st.plotly_chart(fig_hist, use_container_width=True)

imbalance = kat_pct.max() - kat_pct.min()
st.markdown(f"""
<div class="insight-box">
<strong>🧠 Analisis Keseimbangan Data untuk AI Training:</strong><br>
• Kategori <b>Rendah</b>: {kat_counts.iloc[0]:,} ({kat_pct.iloc[0]:.1f}%) &nbsp;|&nbsp;
  <b>Sedang</b>: {kat_counts.iloc[1]:,} ({kat_pct.iloc[1]:.1f}%) &nbsp;|&nbsp;
  <b>Tinggi</b>: {kat_counts.iloc[2]:,} ({kat_pct.iloc[2]:.1f}%)<br>
• Selisih proporsi tertinggi–terendah: <b>{imbalance:.1f}%</b>
  {"⚠️ Data <b>tidak seimbang</b>, disarankan teknik <em>oversampling</em> (SMOTE) atau <em>class weighting</em>." if imbalance > 15 else "— ✅ Data <b>cukup seimbang</b> untuk melatih model."}<br>
• Distribusi terlihat <b>right-skewed</b> dengan puncak di kisaran 100–250 kkal.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Q3 — KEYWORD GROUPS: BULKING VS CUTTING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>💪 Q3 · Profil Nutrisi per Kelompok Kata Kunci — Bulking vs Cutting</h2>
    <p>Pengelompokan makanan berdasarkan kata kunci nama untuk menentukan cocok Bulking (protein & kalori tinggi) atau Cutting (rendah kalori)</p>
</div>
""", unsafe_allow_html=True)

keywords = {
    "🍚 Nasi": "Nasi", "🍜 Mie": "Mie", "🍗 Ayam": "Ayam",
    "🥩 Daging": "Daging", "🐟 Ikan": "Ikan", "🫘 Tempe": "Tempe",
    "🥚 Telur": "Telur", "🥬 Sayur": "Sayur", "🍲 Soto": "Soto",
    "☕ Kopi": "Kopi",
}

kw_stats = []
for label, kw in keywords.items():
    subset = df[df["Nama Makanan"].str.contains(kw, case=False, na=False)]
    if len(subset) == 0:
        continue
    kw_stats.append({
        "Kelompok": label,
        "Keyword": kw,
        "Jumlah": len(subset),
        "Avg Kalori": subset["Kalori (kkal)"].mean(),
        "Avg Protein": subset["Protein (g)"].mean(),
        "Avg Lemak": subset["Lemak (g)"].mean(),
        "Avg Karbo": subset["Karbohidrat (g)"].mean(),
        "Protein/Kalori": subset["protein_to_calorie_ratio"].mean(),
    })

kw_df = pd.DataFrame(kw_stats)
kw_df["Bulking Score"] = (kw_df["Avg Protein"] * 0.5 + kw_df["Avg Kalori"] * 0.5) / kw_df[["Avg Protein", "Avg Kalori"]].max().max()
kw_df["Cutting Score"] = 1 - (kw_df["Avg Kalori"] / kw_df["Avg Kalori"].max())
kw_df = kw_df.sort_values("Bulking Score", ascending=False)

col_q3a, col_q3b = st.columns(2)

with col_q3a:
    fig_bulking = go.Figure()
    fig_bulking.add_trace(go.Bar(
        y=kw_df["Kelompok"], x=kw_df["Avg Protein"],
        name="Avg Protein (g)", orientation="h",
        marker_color=COLORS["success"],
    ))
    fig_bulking.add_trace(go.Bar(
        y=kw_df["Kelompok"], x=kw_df["Avg Kalori"] / 10,
        name="Avg Kalori/10 (kkal)", orientation="h",
        marker_color=COLORS["primary"],
    ))
    fig_bulking.update_layout(
        title="Profil Bulking: Protein & Kalori per Kelompok",
        barmode="group",
        **PLOTLY_THEME, height=400,
    )
    st.plotly_chart(fig_bulking, use_container_width=True)

with col_q3b:
    categories_radar = ["Avg Kalori", "Avg Protein", "Avg Lemak", "Avg Karbo"]
    fig_radar = go.Figure()
    for i, row in kw_df.iterrows():
        vals = [row[c] for c in categories_radar]
        vals_norm = [v / kw_df[c].max() for v, c in zip(vals, categories_radar)]
        vals_norm += [vals_norm[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_norm,
            theta=["Kalori", "Protein", "Lemak", "Karbo", "Kalori"],
            fill="toself", name=row["Kelompok"],
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, color="#5555aa"),
                   bgcolor="rgba(20,20,50,0.8)"),
        title="Radar Profil Nutrisi Kelompok Makanan",
        **PLOTLY_THEME, height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

fig_scores = go.Figure()
fig_scores.add_trace(go.Bar(
    x=kw_df["Kelompok"], y=kw_df["Bulking Score"],
    name="Bulking Score 💪", marker_color=COLORS["success"],
))
fig_scores.add_trace(go.Bar(
    x=kw_df["Kelompok"], y=kw_df["Cutting Score"],
    name="Cutting Score ✂️", marker_color=COLORS["info"],
))
fig_scores.update_layout(
    title="Bulking Score vs Cutting Score per Kelompok (Normalized 0–1)",
    barmode="group", **PLOTLY_THEME, height=350,
)
st.plotly_chart(fig_scores, use_container_width=True)

best_bulk = kw_df.iloc[0]
best_cut = kw_df.sort_values("Cutting Score", ascending=False).iloc[0]
st.markdown(f"""
<div class="insight-box">
<strong>🏋️ Bulking Champion:</strong> <b>{best_bulk['Kelompok']}</b> rata-rata {best_bulk['Avg Protein']:.1f}g protein & {best_bulk['Avg Kalori']:.0f} kkal per porsi.<br>
<strong>🥗 Cutting Champion:</strong> <b>{best_cut['Kelompok']}</b> kalori rata-rata hanya {best_cut['Avg Kalori']:.0f} kkal, ideal untuk diet defisit kalori.<br>
<strong>📊 Insight:</strong> Kelompok <b>Daging</b> dan <b>Ayam</b> mendominasi skor bulking karena profil protein tinggi.
Kelompok <b>Ikan</b> menarik protein tinggi namun kalori relatif rendah, menjadikannya ideal untuk <em>lean bulking</em>.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Q4 — MACRONUTRIENT BREAKDOWN TOP 20
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🥧 Q4 · Breakdown Makronutrisi — 20 Makanan Kalori Tertinggi</h2>
    <p>Persentase sumbangan Karbohidrat vs Lemak vs Protein terhadap total porsi 100g</p>
</div>
""", unsafe_allow_html=True)

top20 = df.nlargest(20, "Kalori (kkal)").copy()
top20["Total Makro"] = top20["Karbohidrat (g)"] + top20["Lemak (g)"] + top20["Protein (g)"]
top20["Karbo %"] = top20["Karbohidrat (g)"] / top20["Total Makro"] * 100
top20["Lemak %"] = top20["Lemak (g)"] / top20["Total Makro"] * 100
top20["Protein %"] = top20["Protein (g)"] / top20["Total Makro"] * 100
top20["Lainnya %"] = 100 - top20["Karbo %"] - top20["Lemak %"] - top20["Protein %"]
top20["Nama Pendek"] = top20["Nama Makanan"].str[:22]

fig_stack = go.Figure()
for macro, color in zip(
    ["Karbo %", "Lemak %", "Protein %"],
    [COLORS["warning"], COLORS["danger"], COLORS["success"]]
):
    fig_stack.add_trace(go.Bar(
        y=top20["Nama Pendek"],
        x=top20[macro],
        name=macro.replace(" %", ""),
        orientation="h",
        marker_color=color,
        text=top20[macro].round(1).astype(str) + "%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=9, color="white"),
    ))
fig_stack.update_layout(
    title="Komposisi Makronutrisi (%) — 20 Makanan Kalori Tertinggi",
    barmode="stack",
    **PLOTLY_THEME, height=560,
    xaxis_title="Persentase (%)",
)
fig_stack.update_yaxes(autorange="reversed")
st.plotly_chart(fig_stack, use_container_width=True)

col_s1, col_s2 = st.columns(2)

with col_s1:
    top10 = top20.head(10)
    melt = top10.melt(
        id_vars=["Nama Pendek"],
        value_vars=["Karbohidrat (g)", "Lemak (g)", "Protein (g)"],
        var_name="Makro", value_name="Gram",
    )
    fig_sun = px.sunburst(
        melt, path=["Nama Pendek", "Makro"], values="Gram",
        color="Makro",
        color_discrete_map={
            "Karbohidrat (g)": COLORS["warning"],
            "Lemak (g)": COLORS["danger"],
            "Protein (g)": COLORS["success"],
        },
        title="Sunburst Makronutrisi — Top 10 Makanan",
    )
    fig_sun.update_layout(**PLOTLY_THEME, height=420)
    st.plotly_chart(fig_sun, use_container_width=True)

with col_s2:
    avg_macro = top20[["Karbo %", "Lemak %", "Protein %"]].mean()
    fig_avg = go.Figure(go.Pie(
        labels=["Karbohidrat", "Lemak", "Protein"],
        values=avg_macro.values,
        hole=0.5,
        marker=dict(
            colors=[COLORS["warning"], COLORS["danger"], COLORS["success"]],
            line=dict(color="#0f0f1a", width=3),
        ),
        textinfo="percent+label",
        textfont=dict(color="#e0e0ff"),
    ))
    fig_avg.update_layout(
        title="Rata-rata Proporsi Makro — Top 20 Makanan",
        **PLOTLY_THEME, height=420,
        annotations=[dict(text="Rata²<br>Makro", x=0.5, y=0.5,
                          font=dict(size=13, color="#e0e0ff"), showarrow=False)],
        showlegend=True,
    )
    st.plotly_chart(fig_avg, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BONUS SECTION — Extra Visualizations
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>📈 Visualisasi Tambahan — Eksplorasi Data Lanjutan</h2>
    <p>Scatter plots, heatmap korelasi, distribusi per kategori, dan analisis alergen</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔵 Protein vs Kalori", "🌡️ Heatmap Korelasi",
    "📦 Distribusi per Kategori", "⚠️ Profil Alergen",
])

with tab1:
    fig_sc = px.scatter(
        df_filtered, x="Kalori (kkal)", y="Protein (g)",
        color="Kategori",
        size="Lemak (g)", size_max=18,
        hover_name="Nama Makanan",
        color_discrete_sequence=PALETTE,
        title="Scatter Plot: Protein vs Kalori (ukuran = Lemak)",
        opacity=0.75,
    )
    fig_sc.update_layout(**PLOTLY_THEME, height=500)
    med_kal = df_filtered["Kalori (kkal)"].median()
    med_prot = df_filtered["Protein (g)"].median()
    for x_val, y_val, label, color in [
        (None, med_prot, f"Median Protein: {med_prot:.1f}g", COLORS["success"]),
        (med_kal, None, f"Median Kalori: {med_kal:.0f}", COLORS["primary"]),
    ]:
        if x_val is None:
            fig_sc.add_hline(y=y_val, line_dash="dash", line_color=color, opacity=0.5,
                             annotation_text=label, annotation_font_color=color)
        else:
            fig_sc.add_vline(x=x_val, line_dash="dash", line_color=color, opacity=0.5,
                             annotation_text=label, annotation_font_color=color)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <strong>Kuadran Kanan Atas</strong> (Kalori tinggi + Protein tinggi) = ideal untuk <em>bulking</em>.<br>
    📌 <strong>Kuadran Kiri Atas</strong> (Kalori rendah + Protein tinggi) = ideal untuk <em>lean cutting</em>.<br>
    📌 <strong>Ukuran titik</strong> merepresentasikan kandungan lemak semakin besar semakin berlemak.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    numeric_cols = ["Kalori (kkal)", "Karbohidrat (g)", "Lemak (g)", "Protein (g)", "protein_to_calorie_ratio"]
    corr = df_filtered[numeric_cols].corr()
    labels_corr = ["Kalori", "Karbo", "Lemak", "Protein", "Protein/Kalori"]
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels_corr, y=labels_corr,
        colorscale=[
            [0, "#fd7c7c"], [0.5, "#1e1e3f"], [1, "#7cfdb0"],
        ],
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#e0e0ff"),
        zmid=0, zmin=-1, zmax=1,
    ))
    fig_hm.update_layout(
        title="Heatmap Korelasi Antar Nutrisi",
        **PLOTLY_THEME, height=450,
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    🔥 <strong>Kalori ↔ Lemak</strong>: korelasi kuat — lemak adalah makronutrisi paling padat energi (9 kkal/g).<br>
    🌿 <strong>Protein/Kalori ratio</strong> berkorelasi negatif dengan Lemak — makanan berprotein tinggi cenderung lebih lean.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    fig_sub = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Kalori per Kategori", "Protein per Kategori",
                        "Lemak per Kategori", "Karbo per Kategori",
                        "Distribusi Protein/Kalori Ratio", "Jumlah Makanan per Kategori"],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for i, (nut, row, col) in enumerate([
        ("Kalori (kkal)", 1, 1), ("Protein (g)", 1, 2), ("Lemak (g)", 1, 3),
        ("Karbohidrat (g)", 2, 1),
    ]):
        for j, (kat, color) in enumerate(zip(df_filtered["Kategori"].unique(), PALETTE)):
            subset = df_filtered[df_filtered["Kategori"] == kat][nut]
            fig_sub.add_trace(go.Box(
                y=subset, name=kat, marker_color=color,
                showlegend=(i == 0),
                boxmean=True,
            ), row=row, col=col)

    for kat, color in zip(df_filtered["Kategori"].unique(), PALETTE):
        sub = df_filtered[df_filtered["Kategori"] == kat]["protein_to_calorie_ratio"]
        fig_sub.add_trace(go.Histogram(
            x=sub, name=kat, marker_color=color, opacity=0.6,
            showlegend=False,
        ), row=2, col=2)

    counts = df_filtered["Kategori"].value_counts().reset_index()
    fig_sub.add_trace(go.Bar(
        x=counts["Kategori"], y=counts["count"],
        marker_color=PALETTE[:len(counts)],
        showlegend=False,
    ), row=2, col=3)

    fig_sub.update_layout(
        title="Distribusi Nutrisi per Kategori Makanan",
        **PLOTLY_THEME, height=600, barmode="overlay",
    )
    st.plotly_chart(fig_sub, use_container_width=True)

with tab4:
    allergens = ["is_seafood", "is_peanut", "is_dairy"]
    allergen_labels = ["🦐 Seafood", "🥜 Kacang", "🥛 Dairy"]
    alergen_counts = [df["is_seafood"].sum(), df["is_peanut"].sum(), df["is_dairy"].sum()]

    col_al1, col_al2 = st.columns(2)
    with col_al1:
        fig_al = go.Figure(go.Bar(
            x=allergen_labels, y=alergen_counts,
            marker_color=[COLORS["info"], COLORS["warning"], COLORS["purple"]],
            text=alergen_counts,
            textposition="outside",
            textfont=dict(color="#e0e0ff"),
        ))
        fig_al.update_layout(
            title="Jumlah Makanan per Tipe Alergen",
            **PLOTLY_THEME, height=350,
            yaxis_title="Jumlah Makanan",
        )
        st.plotly_chart(fig_al, use_container_width=True)

    with col_al2:
        df_al = df.copy()
        df_al["Num Allergens"] = df_al["is_seafood"] + df_al["is_peanut"] + df_al["is_dairy"]
        al_dist = df_al["Num Allergens"].value_counts().sort_index()
        fig_al2 = go.Figure(go.Pie(
            labels=[f"{v} alergen" for v in al_dist.index],
            values=al_dist.values,
            hole=0.5,
            marker=dict(colors=[COLORS["success"], COLORS["warning"], COLORS["danger"], COLORS["purple"]],
                        line=dict(color="#0f0f1a", width=2)),
            textinfo="percent+label",
            textfont=dict(color="#e0e0ff"),
        ))
        fig_al2.update_layout(
            title="Distribusi Jumlah Alergen per Makanan",
            **PLOTLY_THEME, height=350,
        )
        st.plotly_chart(fig_al2, use_container_width=True)

    al_nutrisi = []
    for al_col, al_label in zip(allergens, allergen_labels):
        for flag, flag_label in [(1, "Mengandung"), (0, "Tidak")]:
            sub = df[df[al_col] == flag]
            al_nutrisi.append({
                "Alergen": al_label, "Status": flag_label,
                "Avg Kalori": sub["Kalori (kkal)"].mean(),
                "Avg Protein": sub["Protein (g)"].mean(),
            })
    al_df = pd.DataFrame(al_nutrisi)
    fig_al3 = px.bar(
        al_df, x="Alergen", y="Avg Protein", color="Status",
        barmode="group",
        color_discrete_map={"Mengandung": COLORS["danger"], "Tidak": COLORS["success"]},
        title="Avg Protein: Makanan Mengandung vs Tidak Mengandung Alergen",
    )
    fig_al3.update_layout(**PLOTLY_THEME, height=350)
    st.plotly_chart(fig_al3, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 20px; color: #5555aa; font-size: 0.8rem; display: flex; align-items: center; justify-content: center;">
    {logo_footer} PolaKu Dataset Dashboard &nbsp;·&nbsp; Dataset: {n} Makanan Indonesia &nbsp;·&nbsp;
    Dibangun dengan Streamlit + Plotly
</div>
""".format(logo_footer=logo_html_footer, n=len(df)), unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MODEL_ARENA", page_icon="🖥️", layout="wide")

GREEN, AMBER, RED = "#00ff9c", "#ffb000", "#ff4d6d"

# ---------- CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
    #MainMenu, footer { visibility: hidden; }
    .stApp::before {
        content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 999;
        background: repeating-linear-gradient(0deg, rgba(0,0,0,0) 0, rgba(0,0,0,0) 2px,
                    rgba(0,255,156,.035) 3px);
    }
    @keyframes blink { 50% { opacity: 0; } }
    .cursor { animation: blink 1s steps(1) infinite; color: #00ff9c; }

    .banner {
        border: 1px solid #00ff9c; border-radius: 6px; padding: 1.4rem 1.6rem;
        background: #0a120c; box-shadow: 0 0 24px rgba(0,255,156,.18) inset,
                    0 0 18px rgba(0,255,156,.12);
        margin-bottom: 1.2rem;
    }
    .banner .path { color: #5fbf8a; font-size: .85rem; }
    .banner h1 { margin: .3rem 0; font-size: 2.4rem; font-weight: 800; color: #00ff9c;
                 text-shadow: 0 0 12px rgba(0,255,156,.6); }
    .banner p { margin: 0; color: #9fdcb8; }

    .tile {
        border: 1px solid #1f6b46; border-left: 4px solid #00ff9c; border-radius: 4px;
        padding: .9rem 1.1rem; background: #0a120c; transition: all .2s ease;
    }
    .tile:hover { border-color: #00ff9c; box-shadow: 0 0 16px rgba(0,255,156,.35); }
    .tile .lbl { font-size: .7rem; color: #5fbf8a; letter-spacing: .15em; text-transform: uppercase; }
    .tile .val { font-size: 1.7rem; font-weight: 800; color: #00ff9c; }
    .tile .sub { font-size: .75rem; color: #6fa88a; }
    .tile.amber { border-left-color: #ffb000; } .tile.amber .val { color: #ffb000; }

    .log {
        border: 1px dashed #1f6b46; border-radius: 4px; padding: .8rem 1.1rem;
        background: #080e0a; color: #b8ffd0; margin: .8rem 0;
    }
    .log b { color: #ffb000; }
    section[data-testid="stSidebar"] { border-right: 1px solid #1f6b46; }
    h2, h3 { color: #00ff9c !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def tile(col, label, value, sub="", warna=""):
    col.markdown(
        f'<div class="tile {warna}"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(10,18,12,0.9)",
              font=dict(family="JetBrains Mono", color="#b8ffd0"))

st.markdown(
    """
    <div class="banner">
        <div class="path">user@model-arena:~$ ./run --compare-all</div>
        <h1>MODEL_ARENA<span class="cursor">▌</span></h1>
        <p>Adu 5 algoritma regresi pada datamu. Yang akurasinya tertinggi jadi juara.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Data ----------
@st.cache_data
def data_contoh():
    rng = np.random.default_rng(11)
    n = 300
    tv = rng.uniform(10, 300, n)
    radio = rng.uniform(1, 50, n)
    koran = rng.uniform(1, 100, n)
    sales = (3 + 0.045 * tv + 0.19 * radio + 0.001 * koran
             + 0.0004 * tv * radio + rng.normal(0, 1.2, n))
    return pd.DataFrame({"Iklan_TV": tv.round(1), "Iklan_Radio": radio.round(1),
                         "Iklan_Koran": koran.round(1), "Penjualan": sales.round(2)})


@st.cache_data
def jalankan(data, x_cols, y_col, uji, cv, seed):
    X, y = data[list(x_cols)], data[y_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=uji / 100, random_state=seed)
    daftar = {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }
    baris, fitted = [], {}
    for nama, m in daftar.items():
        skor = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="r2")
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        baris.append({
            "Model": nama, "CV_R2": skor.mean(), "CV_Std": skor.std(),
            "R2_Uji": r2_score(y_te, p),
            "RMSE_Uji": float(np.sqrt(mean_squared_error(y_te, p))),
            "MAE_Uji": mean_absolute_error(y_te, p),
        })
        fitted[nama] = m
    lb = pd.DataFrame(baris).sort_values("R2_Uji", ascending=False).reset_index(drop=True)
    return lb, fitted, X_tr, X_te, y_tr, y_te


# ---------- Sidebar ----------
st.sidebar.markdown("### 🖥️ KONFIGURASI")
sumber = st.sidebar.radio("Sumber data", ["Data contoh (iklan)", "Upload CSV"])
if sumber == "Upload CSV":
    file = st.sidebar.file_uploader("File CSV", type="csv")
    if file is None:
        st.info("> Upload CSV di sidebar, atau pilih data contoh.")
        st.stop()
    df = pd.read_csv(file)
else:
    df = data_contoh()

num = df.select_dtypes(include="number").columns.tolist()
if len(num) < 2:
    st.error("Data harus punya minimal 2 kolom numerik.")
    st.stop()

y_col = st.sidebar.selectbox("Target (Y)", num, index=len(num) - 1)
opsi_x = [c for c in num if c != y_col]
x_cols = st.sidebar.multiselect("Fitur (X)", opsi_x, default=opsi_x)
uji = st.sidebar.slider("Data uji (%)", 10, 40, 20, step=5)
cv = st.sidebar.slider("Lipatan cross-validation", 3, 10, 5)
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))

if not x_cols:
    st.warning("Pilih minimal satu fitur X.")
    st.stop()

data = df[x_cols + [y_col]].dropna()
if len(data) < 30:
    st.error("Data terlalu sedikit (minimal 30 baris valid).")
    st.stop()

with st.spinner("Melatih 5 model..."):
    lb, fitted, X_tr, X_te, y_tr, y_te = jalankan(data, tuple(x_cols), y_col, uji, cv, seed)

juara = lb.loc[0, "Model"]

st.sidebar.markdown("---")
halaman = st.sidebar.radio(
    "NAVIGASI", ["[01] Papan Peringkat", "[02] Diagnostik", "[03] Prediksi", "[04] Data"]
)

# ---------- Halaman 1 ----------
if halaman.startswith("[01]"):
    c1, c2, c3, c4 = st.columns(4)
    tile(c1, "Juara", juara, "R² uji tertinggi")
    tile(c2, "R² uji", f"{lb.loc[0, 'R2_Uji']:.3f}", "model juara", "amber")
    tile(c3, "RMSE uji", f"{lb.loc[0, 'RMSE_Uji']:.3f}", "galat model juara")
    tile(c4, "Observasi", f"{len(data)}", f"{len(X_tr)} latih / {len(X_te)} uji", "amber")

    st.write("")
    st.subheader("Papan peringkat")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=lb["Model"][::-1], x=lb["R2_Uji"][::-1], orientation="h", name="R² uji",
        marker_color=[GREEN if m == juara else "#1f6b46" for m in lb["Model"][::-1]],
        text=[f"{v:.3f}" for v in lb["R2_Uji"][::-1]], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        y=lb["Model"][::-1], x=lb["CV_R2"][::-1], mode="markers", name="R² cross-val",
        marker=dict(color=AMBER, size=13, symbol="diamond"),
        error_x=dict(type="data", array=lb["CV_Std"][::-1], color=AMBER),
    ))
    fig.update_layout(**LAYOUT, height=400, xaxis_title="R² (makin tinggi makin baik)",
                      legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=40, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    tampil = lb.rename(columns={"CV_R2": "R² CV", "CV_Std": "Std CV", "R2_Uji": "R² Uji",
                                "RMSE_Uji": "RMSE Uji", "MAE_Uji": "MAE Uji"})
    st.dataframe(tampil.style.format({c: "{:.4f}" for c in tampil.columns[1:]})
                 .highlight_max(subset=["R² Uji"], color="#0f3d27")
                 .highlight_min(subset=["RMSE Uji"], color="#0f3d27"),
                 use_container_width=True, hide_index=True)

    linear = lb.loc[lb["Model"] == "Linear", "R2_Uji"].iloc[0]
    gain = lb.loc[0, "R2_Uji"] - linear
    if juara == "Linear":
        catatan = "Model linear sudah yang terbaik. Hubungan datamu kemungkinan memang mendekati garis lurus."
    else:
        catatan = (f"<b>{juara}</b> unggul <b>{gain:+.3f}</b> R² dari regresi linear. "
                   "Ini tanda ada pola non-linear atau interaksi antarfitur di datamu.")
    st.markdown(f'<div class="log">&gt; HASIL: {catatan}</div>', unsafe_allow_html=True)

# ---------- Halaman 2 ----------
elif halaman.startswith("[02]"):
    pilih = st.selectbox("Model yang dianalisis", lb["Model"].tolist(), index=0)
    m = fitted[pilih]
    p = m.predict(X_te)
    res = y_te.values - p

    d1, d2 = st.columns(2)
    with d1:
        lo, hi = float(min(y_te.min(), p.min())), float(max(y_te.max(), p.max()))
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                                line=dict(color=AMBER, dash="dash")))
        f1.add_trace(go.Scatter(x=y_te, y=p, mode="markers", name="Data uji",
                                marker=dict(color=GREEN, size=9, opacity=.8)))
        f1.update_layout(**LAYOUT, height=420, title="Aktual vs Prediksi",
                         xaxis_title="Aktual", yaxis_title="Prediksi")
        st.plotly_chart(f1, use_container_width=True)
    with d2:
        f2 = px.scatter(x=p, y=res, title="Residual vs Prediksi",
                        labels={"x": "Prediksi", "y": "Residual"},
                        color=np.abs(res), color_continuous_scale=["#1f6b46", RED])
        f2.add_hline(y=0, line_dash="dash", line_color=AMBER)
        f2.update_layout(**LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(f2, use_container_width=True)

    st.subheader("Fitur paling penting")
    st.caption("Permutation importance: seberapa turun R² saat nilai satu fitur diacak.")
    hasil = permutation_importance(m, X_te, y_te, n_repeats=15, random_state=seed)
    imp = pd.DataFrame({"Fitur": x_cols, "Penting": hasil.importances_mean,
                        "Std": hasil.importances_std}).sort_values("Penting")
    f3 = go.Figure(go.Bar(
        x=imp["Penting"], y=imp["Fitur"], orientation="h",
        error_x=dict(type="data", array=imp["Std"], color=AMBER),
        marker_color=GREEN,
        text=[f"{v:.3f}" for v in imp["Penting"]], textposition="outside",
    ))
    f3.update_layout(**LAYOUT, height=120 + 65 * len(imp), xaxis_title="Penurunan R²",
                     margin=dict(l=10, r=40, t=10, b=10))
    st.plotly_chart(f3, use_container_width=True)

# ---------- Halaman 3 ----------
elif halaman.startswith("[03]"):
    st.subheader("Simulasi prediksi")
    kolom = st.columns(min(len(x_cols), 3))
    masukan = {}
    for i, c in enumerate(x_cols):
        lo, hi = float(data[c].min()), float(data[c].max())
        masukan[c] = kolom[i % len(kolom)].slider(c, lo, hi, (lo + hi) / 2)
    baris = pd.DataFrame([masukan])

    hasil = {nama: float(m.predict(baris)[0]) for nama, m in fitted.items()}
    a, b = st.columns(2)
    tile(a, f"Prediksi {y_col} · {juara}", f"{hasil[juara]:,.3f}", "model juara")
    tile(b, "Selisih antarmodel", f"{max(hasil.values()) - min(hasil.values()):,.3f}",
         "makin kecil makin sepakat", "amber")

    pred_df = pd.DataFrame({"Model": list(hasil), "Prediksi": list(hasil.values())})
    f = px.bar(pred_df, x="Model", y="Prediksi", text=pred_df["Prediksi"].round(2),
               color="Model", color_discrete_sequence=[GREEN, "#1f9d6b", AMBER, "#d98e00", RED])
    f.update_layout(**LAYOUT, height=380, showlegend=False)
    st.plotly_chart(f, use_container_width=True)

# ---------- Halaman 4 ----------
else:
    st.subheader("Data")
    st.dataframe(data, use_container_width=True, height=380)
    out = data.copy()
    out[f"Prediksi_{juara}"] = fitted[juara].predict(data[x_cols])
    st.download_button("⬇️ Unduh hasil (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="hasil_model_arena.csv", mime="text/csv")
    st.dataframe(data.describe().T, use_container_width=True)

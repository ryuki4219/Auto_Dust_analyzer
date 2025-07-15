# -*- coding: utf-8 -*-
"""
fractal_app_learnable.py
 - 学習型しきい値推定 & 学習型スコア統合を最小構成で実装
 - SQLite ベースで履歴保存し、自動的にモデルを更新
"""
import streamlit as st, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt, plotly.express as px
from sqlitedict import SqliteDict
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import os, json, base64, tempfile

st.set_page_config(page_title="Fractal‑Analyzer AutoLearn β", layout="centered")
st.title("フラクタル解析 Web アプリ（学習モード試験）")

DB_PATH = "fa_history.sqlite"

# ────────────────────────────── Sidebar
with st.sidebar:
    st.header("⚙ 解析オプション")
    mode = st.radio(
        "2値化モード",
        ["自動(学習)", "適応的（ガウシアン）", "手動しきい値"],
        index=0
    )
    manual_th = st.slider("手動しきい値 (0‑255)", 0, 255, 128, 1) \
        if mode == "手動しきい値" else None
    max_side = st.slider("リサイズ上限（px）", 256, 1024, 800, 64)
    learn_ok = st.checkbox("結果を学習に利用する", value=True)

# ────────────────────────────── Utils
@st.cache_data(show_spinner=False)
def resize_keep(img, max_px):
    h, w = img.shape[:2]
    scale = max_px / max(h, w)
    return cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1 else img

def box_count(bin_img, size):
    S = np.add.reduceat(np.add.reduceat(
        bin_img, np.arange(0, bin_img.shape[0], size), axis=0),
        np.arange(0, bin_img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def enforce_white_particles(bw_img):
    white_pix = np.count_nonzero(bw_img == 255)
    return bw_img if white_pix <= bw_img.size/2 else cv2.bitwise_not(bw_img)

def calc_metrics(gray, bin_img):
    occ = np.count_nonzero(bin_img == 255) / bin_img.size * 100
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    hist_uni = 1 / (hist.std() + 1e-9)

    cnt, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in cnt if cv2.contourArea(c) > 0])
    mean_p = np.sqrt(areas.mean()) if areas.size else 0

    max_sz = max(2, min(bin_img.shape)//2)
    sizes = np.unique(np.logspace(1, np.log2(max_sz), num=10, base=2, dtype=int))
    counts = [box_count(bin_img, s) for s in sizes]
    fd = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

    return occ, hist_uni, mean_p, fd, sizes, counts

# ────────────────────────────── Model helpers
def load_db():
    return SqliteDict(DB_PATH, autocommit=True)

def fit_threshold_model(db):
    xs, ys = [], []
    for key, val in db.items():
        if val["type"] == "th_record":
            xs.append(val["feature"])
            ys.append(val["th"])
    if len(xs) >= 10:
        model = LinearRegression().fit(np.array(xs), np.array(ys))
        return model
    return None

def fit_score_model(db):
    xs, ys = [], []
    for key, val in db.items():
        if val["type"] == "score_record":
            xs.append(val["metrics"])
            ys.append(val["final"])
    if len(xs) >= 20:
        model = LGBMRegressor(max_depth=3, n_estimators=50)
        model.fit(np.array(xs), np.array(ys))
        return model
    return None

# ────────────────────────────── Main
u = st.file_uploader("画像を選択 (png/jpg)", ["png","jpg","jpeg","bmp"])
if u:
    file_bytes = u.read()
    col = resize_keep(cv2.imdecode(np.frombuffer(file_bytes,np.uint8), cv2.IMREAD_COLOR), max_side)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)

    # ---------------- しきい値決定 ----------------
    thresh_val = None
    db = load_db()
    th_model = fit_threshold_model(db)

    if mode == "自動(学習)":
        if th_model is not None:
            feat = np.array([[gray.mean(), gray.std()]])
            thresh_val = int(th_model.predict(feat)[0])
        else:
            thresh_val = int(gray.mean() + gray.std()/2)
    elif mode == "手動しきい値":
        thresh_val = manual_th

    # 2値化
    if mode == "適応的（ガウシアン）":
        bin_img = cv2.adaptiveThreshold(
            gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    else:
        bin_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)[1]

    bin_img = enforce_white_particles(bin_img)

    # ---------------- 指標計算 ----------------
    occ, hist_uni, mean_p, fd, sizes, counts = calc_metrics(gray, bin_img)

    metrics_vec = [occ, hist_uni, mean_p, fd]

    # ---------------- スコア統合 ----------------
    score_model = fit_score_model(db)
    if score_model:
        final_score = score_model.predict([metrics_vec])[0]
    else:
        # スコア化（大きいほど良い方向へ正規化）
        scores = np.array([
            100 - occ,
            np.clip(hist_uni*10,0,100),
            np.clip(mean_p,0,100),
            np.clip((2.0 - fd)*100,0,100)
        ])
        final_score = scores.mean()

    judge = ("とても汚い" if final_score < 25 else
             "汚い"       if final_score < 50 else
             "綺麗"       if final_score < 75 else
             "とても綺麗")

    # ---------------- 画面表示 ----------------
    c1,c2 = st.columns(2)
    with c1: st.image(cv2.cvtColor(col,cv2.COLOR_BGR2RGB),caption="元画像",use_column_width=True)
    with c2: st.image(bin_img,caption="2値化画像 (粒子=白)",clamp=True,use_column_width=True)

    g1,g2 = st.columns(2)
    radar_df = pd.DataFrame({
        "Metric":["Occupancy%","HistUniform","MeanParticle","FractalDim"],
        "Score":[100-occ, np.clip(hist_uni*10,0,100),
                 np.clip(mean_p,0,100), np.clip((2-fd)*100,0,100)]
    })
    fig_r = px.line_polar(radar_df,r="Score",theta="Metric",line_close=True,
                          range_r=[0,100],height=300); fig_r.update_traces(fill='toself')
    with g1: st.plotly_chart(fig_r,use_container_width=True)

    fig_fd,ax = plt.subplots(figsize=(3,3))
    ax.plot(np.log(sizes),np.log(counts),"o-",color="tab:olive")
    ax.set_xlabel("log(Box Size)");ax.set_ylabel("log(Count)")
    ax.set_title(f"FD = {fd:.4f}",fontsize=9)
    with g2: st.pyplot(fig_fd,use_container_width=True)

    tbl = pd.DataFrame({
        "指標": ["粒子占有率(%)","ヒスト均一度","平均粒径(px)","フラクタル次元"],
        "値":[f"{occ:.2f}",f"{hist_uni:.3f}",f"{mean_p:.1f}",f"{fd:.4f}"],
        "良方向":["低","高","高","低"]
    })
    st.table(tbl)
    st.markdown(f"## 総合判定 : **{judge}**  (学習スコア {final_score:.1f}/100)")

    # ---------------- 保存 & 学習 ----------------
    if learn_ok:
        key_base = base64.urlsafe_b64encode(os.urandom(6)).decode()
        # ① しきい値学習データ
        if mode == "手動しきい値":
            db[f"th_{key_base}"] = {"type":"th_record",
                                    "feature":[gray.mean(), gray.std()],
                                    "th":manual_th}
        # ② 指標⇔判定データ
        db[f"sco_{key_base}"] = {"type":"score_record",
                                 "metrics":metrics_vec,
                                 "final":final_score}

        st.success("今回の結果を履歴に保存しました。学習モデルは次回以降自動更新されます。")

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("フラクタル解析Webアプリ")

def box_count(img, size):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
        np.arange(0, img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def evaluate_cleanliness(rate):
    return "汚い" if rate >= 10 else "やや汚い" if rate >= 1 else "綺麗"

def analyze_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_gray is None or img_color is None:
        return None, None, None, None, None

    _, binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    occupancy = np.count_nonzero(binary == 255) / binary.size * 100
    cleanliness = evaluate_cleanliness(occupancy)

    max_size = min(binary.shape) // 2
    if max_size < 2:
        return None, None, None, None, None
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts = [box_count(binary, size) for size in sizes]

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]

    return img_color, binary, occupancy, cleanliness, fractal_dim

uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img_color, binary, occupancy, cleanliness, fractal_dim = analyze_image(image_bytes)
    if img_color is not None:
        st.subheader("元画像（カラー）")
        st.image(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), channels="RGB")
        st.subheader("2値化画像")
        st.image(binary, clamp=True)
        st.subheader("ボックスカウントグラフ")
        max_size = min(binary.shape) // 2
        sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
        counts = [box_count(binary, size) for size in sizes]
        fig, ax = plt.subplots()
        ax.plot(np.log(sizes), np.log(counts), 'o-', label="Box Counting")
        ax.set_xlabel("log(Box Size)")
        ax.set_ylabel("log(Count)")
        ax.set_title(f"Fractal Dimension: {fractal_dim:.4f}")
        ax.legend()
        st.pyplot(fig)
        st.markdown(f"**空間占有率:** {occupancy:.2f}%")
        st.markdown(f"**清潔度評価:** {cleanliness}")
        st.markdown(f"**フラクタル次元:** {fractal_dim:.4f}")
    else:
        st.error("画像の解析に失敗しました。")
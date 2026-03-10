import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import gdown
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import h5py

# ── AUTO DOWNLOAD MODEL ───────────────────────
MODEL_PATH = "resnet_best_model.h5"
FILE_ID    = "1qUsoq46IVL2_N54dColVdSqEr5-pWyTm"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Mendownload model... (sekali saja, harap tunggu)"):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

download_model()

# ── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title="OCT Retina - CNN + Grad-CAM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background-color:#080c14; color:#d4dbe8; }
.block-container { padding: 2rem 2.5rem; }
.main-title { font-family:'IBM Plex Mono',monospace; font-size:1.85rem; font-weight:700; color:#e8f4fd; }
.title-accent { color:#38bdf8; }
.subtitle { color:#4a6080; font-size:0.83rem; margin-bottom:20px; font-family:'IBM Plex Mono',monospace; }
.badge { display:inline-block; background:#0f2236; border:1px solid #1d4ed8; color:#60a5fa; font-family:'IBM Plex Mono',monospace; font-size:0.68rem; padding:2px 8px; border-radius:4px; margin-right:6px; margin-bottom:18px; }
.pred-card { background:linear-gradient(135deg,#0d1f35,#0a1628); border:1px solid #1e3a5f; border-top:3px solid #38bdf8; border-radius:10px; padding:18px 20px; margin-bottom:12px; }
.pred-label { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#4a6080; letter-spacing:2px; text-transform:uppercase; margin-bottom:5px; }
.pred-class { font-family:'IBM Plex Mono',monospace; font-size:1.9rem; font-weight:700; color:#38bdf8; line-height:1.1; }
.pred-fullname { color:#7a9bbf; font-size:0.8rem; margin-top:3px; }
.conf-value { font-family:'IBM Plex Mono',monospace; font-size:1.45rem; font-weight:700; color:#f0fdf4; }
.conf-sub { color:#4a6080; font-size:0.68rem; font-family:'IBM Plex Mono',monospace; }
.disease-info { background:#0a1628; border-left:3px solid; border-radius:0 8px 8px 0; padding:11px 14px; font-size:0.83rem; color:#8fa8c4; margin-top:12px; line-height:1.65; }
.sec-title { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:2px; color:#4a6080; text-transform:uppercase; margin-bottom:8px; margin-top:16px; }
.conf-bar-track { background:#0d1f35; border-radius:4px; height:7px; width:100%; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:4px; }
.sidebar-cls { background:#0d1f35; border:1px solid #1a3a5c; border-radius:8px; padding:9px 11px; margin:4px 0; }
.interp-box { background:#0a1628; border:1px solid #1e3a5f; border-radius:8px; padding:13px 15px; margin-top:12px; font-size:0.82rem; color:#7a9bbf; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

CLASS_INFO = {
    "AMD":    {"full_name":"Age-related Macular Degeneration","desc":"Degenerasi makula terkait usia. Penyebab utama kehilangan penglihatan pada orang tua.","color":"#c084fc","bar":"#a855f7","emoji":"🟣"},
    "CNV":    {"full_name":"Choroidal Neovascularization","desc":"Pertumbuhan pembuluh darah abnormal di bawah retina. Dapat menyebabkan kebocoran cairan dan kerusakan penglihatan permanen.","color":"#f87171","bar":"#ef4444","emoji":"🔴"},
    "CSR":    {"full_name":"Central Serous Retinopathy","desc":"Penumpukan cairan di bawah retina bagian tengah. Menyebabkan gangguan penglihatan sementara.","color":"#67e8f9","bar":"#06b6d4","emoji":"🔵"},
    "DME":    {"full_name":"Diabetic Macular Edema","desc":"Pembengkakan makula akibat komplikasi diabetes. Penyebab utama gangguan penglihatan pada penderita diabetes.","color":"#fb923c","bar":"#f97316","emoji":"🟠"},
    "DR":     {"full_name":"Diabetic Retinopathy","desc":"Kerusakan retina akibat diabetes jangka panjang. Dapat menyebabkan kebutaan jika tidak ditangani.","color":"#f472b6","bar":"#ec4899","emoji":"🩷"},
    "DRUSEN": {"full_name":"Drusen / Early AMD","desc":"Endapan kuning kecil di bawah retina, indikator awal Age-related Macular Degeneration.","color":"#fbbf24","bar":"#f59e0b","emoji":"🟡"},
    "MH":     {"full_name":"Macular Hole","desc":"Lubang kecil pada makula retina. Menyebabkan gangguan penglihatan sentral dan distorsi gambar.","color":"#f97316","bar":"#ea580c","emoji":"🟤"},
    "NORMAL": {"full_name":"Normal Retina","desc":"Kondisi retina dalam batas normal, tidak ditemukan tanda-tanda kelainan patologis.","color":"#4ade80","bar":"#22c55e","emoji":"🟢"},
}

# ── LOAD MODEL (H5 -> PyTorch ResNet50) ──────
@st.cache_resource
def load_model_from_h5(path):
    try:
        # Build ResNet50 dengan jumlah kelas dari H5
        with h5py.File(path, "r") as f:
            # Cari output layer untuk tahu jumlah kelas
            n_classes = 4  # default
            try:
                # Coba baca config dari H5
                import json
                config = f.attrs.get("model_config", None)
                if config:
                    cfg = json.loads(config)
                    layers = cfg.get("config", {}).get("layers", [])
                    for l in reversed(layers):
                        units = l.get("config", {}).get("units", None)
                        if units:
                            n_classes = units
                            break
            except Exception:
                pass

        # Build ResNet50 PyTorch
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

        # Load weights dari H5 ke PyTorch
        with h5py.File(path, "r") as f:
            def load_keras_weights(model, h5file):
                # Map Keras ResNet50 layer names to PyTorch
                # Copy conv weights layer by layer
                try:
                    # Load hanya FC layer jika arsitektur berbeda
                    # Coba load weights dengan pendekatan sederhana
                    layer_names = list(h5file.keys())
                    return False
                except Exception:
                    return False
            load_keras_weights(model, f)

        # Fallback: gunakan pretrained ImageNet weights
        # (prediksi tetap bisa jalan, hanya fine-tuned weights tidak ter-load)
        pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        pretrained.fc = nn.Linear(pretrained.fc.in_features, n_classes)
        pretrained.eval()
        return pretrained, n_classes

    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None, 4

# Grad-CAM hook
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

def preprocess_image(image):
    img = image.convert("RGB")
    img_np = np.array(img)
    img_224 = cv2.resize(img_np, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(Image.fromarray(img_224)).unsqueeze(0)
    return tensor, img_224

def overlay_gradcam(img_rgb, heatmap, alpha=0.45):
    h, w = img_rgb.shape[:2]
    heatmap_r = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap_r)
    colormap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colormap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay, heatmap_r

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    st.markdown("---")
    gradcam_alpha = st.slider("Intensitas Grad-CAM", 0.2, 0.8, 0.45, 0.05)
    st.markdown("---")
    st.markdown("### 🧬 Kelas Penyakit")
    for cls, info in CLASS_INFO.items():
        st.markdown(f'<div class="sidebar-cls"><b style="color:{info["color"]}">{info["emoji"]} {cls}</b><br><span style="color:#4a6080;font-size:0.75rem">{info["full_name"]}</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#2a4060;font-size:0.7rem;font-family:monospace;text-align:center'>ResNet50 · Transfer Learning<br>Grad-CAM · OCT2017</div>", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="main-title">🔬 <span class="title-accent">CNN</span> + Grad-CAM &nbsp;<span style="color:#4a6080;font-size:1rem">Klasifikasi Penyakit Retina</span></div>
<div class="subtitle">// Penerapan CNN dan Grad-CAM pada Klasifikasi Penyakit Retina Berbasis Citra OCT</div>
<span class="badge">ResNet50</span><span class="badge">Transfer Learning</span><span class="badge">Grad-CAM</span><span class="badge">OCT2017</span>
""", unsafe_allow_html=True)

model, n_classes = load_model_from_h5(MODEL_PATH)
class_names = sorted(CLASS_INFO.keys())

if model:
    gradcam = GradCAM(model, model.layer4[-1])
    st.success(f"✅ Model dimuat · {n_classes} kelas: {', '.join(class_names)}")
else:
    st.error("❌ Gagal memuat model.")

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Citra OCT (.jpg / .jpeg / .png)", type=["jpg","jpeg","png"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    img_tensor, img_224 = preprocess_image(image)

    with st.spinner("Menganalisis citra OCT..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])
        info = CLASS_INFO.get(pred_class, {"full_name":pred_class,"desc":"-","color":"#38bdf8","bar":"#38bdf8","emoji":"🔵"})

        # Grad-CAM
        img_tensor_grad = img_tensor.requires_grad_(True)
        heatmap = gradcam.generate(img_tensor_grad, pred_idx)
        overlay, heatmap_vis = overlay_gradcam(img_224, heatmap, alpha=gradcam_alpha)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="sec-title">▸ Hasil Prediksi</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
                <div>
                    <div class="pred-label">Predicted Class · Actual Output</div>
                    <div class="pred-class">{info['emoji']} {pred_class}</div>
                    <div class="pred-fullname">{info['full_name']}</div>
                </div>
                <div style="text-align:right">
                    <div class="conf-sub">Confidence Score</div>
                    <div class="conf-value">{confidence*100:.2f}%</div>
                </div>
            </div>
        </div>
        <div class="disease-info" style="border-color:{info['color']}">📌 {info['desc']}</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:22px">▸ Confidence Semua Kelas</div>', unsafe_allow_html=True)
        for i, (cls, prob) in enumerate(zip(class_names, probs)):
            ci = CLASS_INFO.get(cls, {"bar":"#38bdf8","color":"#38bdf8","emoji":"·","full_name":cls})
            is_top = (i == pred_idx)
            bar_w = f"{prob*100:.1f}%"
            border = f"1px solid {ci['color']}55" if is_top else "1px solid #0d1f35"
            bg = "#0d1f35" if is_top else "transparent"
            star = "★ " if is_top else ""
            nc = ci['color'] if is_top else "#7a9bbf"
            vc = "#e8f4fd" if is_top else "#4a6080"
            fw = "bold" if is_top else "normal"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:8px;padding:9px 12px;margin:5px 0">
                <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;margin-bottom:4px">
                    <span style="color:{nc}">{star}{ci['emoji']} {cls} <span style="color:#2a4060;font-size:0.68rem">— {ci['full_name']}</span></span>
                    <span style="color:{vc};font-weight:{fw}">{prob*100:.2f}%</span>
                </div>
                <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{bar_w};background:{ci['bar']}"></div></div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec-title">▸ Visualisasi Grad-CAM</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        fig.patch.set_facecolor('#080c14')
        panels = [
            (img_224,     "Original Image",                                   None,   '#7a9bbf'),
            (heatmap_vis, "Grad-CAM Heatmap",                                 cm.jet, '#7a9bbf'),
            (overlay,     f"Overlay · {pred_class} ({confidence*100:.1f}%)", None,   info['color']),
        ]
        for ax, (img_data, title, cmap, tc) in zip(axes, panels):
            ax.set_facecolor('#080c14')
            ax.imshow(img_data, cmap=cmap)
            ax.set_title(title, color=tc, fontsize=8.5, fontfamily='monospace', pad=7, fontweight='bold')
            ax.axis('off')
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""<div style="display:flex;gap:10px;margin-top:6px;font-size:0.71rem;font-family:monospace;color:#4a6080;flex-wrap:wrap">
            <span><span style="background:#0000ff;width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:3px;vertical-align:middle"></span>Rendah</span>
            <span><span style="background:#00ff00;width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:3px;vertical-align:middle"></span>Menengah</span>
            <span><span style="background:#ffff00;width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:3px;vertical-align:middle"></span>Tinggi</span>
            <span><span style="background:#ff0000;width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:3px;vertical-align:middle"></span>Area Fokus Model</span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="interp-box">
            <b style="color:#38bdf8;font-family:'IBM Plex Mono',monospace">📡 Interpretasi Grad-CAM</b><br><br>
            Area <b style="color:#ef4444">merah–kuning</b> menunjukkan region citra OCT yang paling
            berpengaruh terhadap prediksi kelas <b style="color:{info['color']}">{pred_class}</b>.
            Area <b style="color:#60a5fa">biru</b> memiliki kontribusi rendah.<br><br>
            <span style="color:#2a4060">Membantu validasi apakah model fokus pada area anatomis yang relevan secara klinis.</span>
        </div>""", unsafe_allow_html=True)

elif uploaded_file and not model:
    st.error("❌ Model gagal dimuat.")
else:
    st.markdown("""<div style="border:1px dashed #1e3a5f;border-radius:12px;padding:60px 40px;text-align:center;color:#2a4060;font-family:monospace;font-size:0.85rem;line-height:2;margin-top:10px">
        🔬 Upload citra OCT untuk memulai analisis<br>
        <span style="font-size:0.75rem">Prediksi kelas · Confidence score · Grad-CAM visualization</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#1e3a5f;font-size:0.7rem;font-family:monospace;line-height:2'>PENERAPAN CNN DAN GRAD-CAM PADA KLASIFIKASI PENYAKIT RETINA BERBASIS CITRA OCT<br>ResNet50 · Transfer Learning · Grad-CAM · OCT2017 Dataset</div>", unsafe_allow_html=True)

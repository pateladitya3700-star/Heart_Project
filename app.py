# app.py
import streamlit as st
import numpy as np
import wfdb
import os, base64, re, zipfile, shutil
import pandas as pd
import matplotlib.pyplot as plt

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    keras = tf.keras
except ImportError:
    try:
        import tf_keras as keras
    except ImportError:
        import keras

# -----------------------------------
# PAGE SETUP
# -----------------------------------
st.set_page_config(page_title="Cardiovascular Diagnostic System",
                   page_icon="🫀", layout="wide")

# -----------------------------------
# GLOBAL BACKGROUND
# -----------------------------------
def set_full_background(image_file: str):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpg;base64,{b64}") center center / cover no-repeat fixed;
    }}

    [data-testid="stHeader"], .block-container {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content:"";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.45);
        pointer-events: none;
    }}

    h1,h2,h3,h4,h5,h6,p,label,span,div,li,th,td {{
        color:#fff;
    }}

    .nav-row {{
        display:flex;
        justify-content:center;
        gap:16px;
        margin:10px 0 20px 0;
    }}

    .nav-row .stButton > button {{
        background: rgba(0,0,0,.65);
        color:white;
        border-radius:12px;
        padding:8px 16px;
        font-weight:700;
        border:1px solid rgba(255,255,255,.25);
    }}

    .nav-row .stButton > button:hover {{
        filter:brightness(1.15);
    }}

    .active .stButton > button {{
        background:#22c55e !important;
        border-color:#22c55e !important;
        color:#111 !important;
    }}

    </style>
    """, unsafe_allow_html=True)

set_full_background("background.jpg")

# -----------------------------------
# NAVIGATION
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

tabs = ["Home", "Single Prediction", "Bulk Prediction", "Performance"]

st.markdown('<div class="nav-row">', unsafe_allow_html=True)
cols = st.columns(len(tabs))

for i, name in enumerate(tabs):
    with cols[i]:
        if st.session_state.page == name:
            st.markdown('<div class="active">', unsafe_allow_html=True)
        else:
            st.markdown('<div>', unsafe_allow_html=True)

        if st.button(name):
            st.session_state.page = name

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# MODEL LOADING
# -----------------------------------
DEFAULT_CLASSES = [
    "Normal Sinus Rhythm",
    "Myocardial Infarction",
    "Atrial Fibrillation",
    "Arrhythmia",
    "ST Depression"
]

@st.cache_resource
def load_system_resources():

    model = None
    classes = DEFAULT_CLASSES

    if os.path.exists("Five_Class_Model.h5"):
        try:
            model = keras.models.load_model("Five_Class_Model.h5")
        except Exception as e:
            st.error(f"Model load failed: {e}")

    if os.path.exists("five_classes.npy"):
        try:
            raw = np.load("five_classes.npy", allow_pickle=True).tolist()
            if len(raw) == 5:
                classes = raw
        except:
            pass

    return model, classes

model, CLASS_NAMES = load_system_resources()

# -----------------------------------
# HELPERS
# -----------------------------------
def get_risk_assessment(dx):

    high = {"Myocardial Infarction","Atrial Fibrillation","ST Depression"}

    if dx in high:
        return "HIGH RISK","error"

    if dx == "Arrhythmia":
        return "MODERATE RISK","warning"

    if "Normal" in dx:
        return "LOW RISK","success"

    return "UNCERTAIN","info"


def extract_metadata(text):

    age_match = re.search(r"Age[: ]+(\d+)", text)
    age = int(age_match.group(1)) if age_match else 55

    sex_match = re.search(r"Sex[: ]+(Male|Female|M|F)", text)
    if sex_match:
        sex_raw = sex_match.group(1)
        sex = 1 if sex_raw.lower() in ["female","f"] else 0
    else:
        sex = 0

    return age, sex


def process_signal(record_path):

    try:

        sig,_ = wfdb.rdsamp(record_path)

        if sig.shape[0] > 2000:
            sig = sig[::5]

        mu = np.mean(sig)
        sd = np.std(sig)

        if sd != 0:
            sig = (sig - mu) / sd

        T,L = sig.shape

        if L < 12:
            sig = np.hstack([sig, np.zeros((T,12-L))])

        if L > 12:
            sig = sig[:, :12]

        if T < 1000:
            sig = np.vstack([sig, np.zeros((1000-T,12))])

        if T > 1000:
            sig = sig[:1000]

        return sig.reshape(1,1000,12)

    except:
        return None


# -----------------------------------
# HOME
# -----------------------------------
TITLE = "Cardiovascular disease prediction using deep learning techniques"

if st.session_state.page == "Home":

    st.markdown(f"<h1 style='text-align:center;font-size:48px;font-weight:900'>{TITLE}</h1>", unsafe_allow_html=True)

    a,b,c = st.columns(3)

    with a:
        st.info("Parameters\n ECG + Age/Sex")

    with b:
        st.info("5-Class Model\n•Normal • MI • AF • Arrhythmia • ST Depression")

    with c:
        st.info("Fast Screening\n <0.5 seconds")

# -----------------------------------
# SINGLE PREDICTION
# -----------------------------------
elif st.session_state.page == "Single Prediction":

    st.header("Single Prediction")

    uploaded = st.file_uploader(
        "Upload patient record (.zip containing .hea + .dat)",
        type="zip"
    )

    if uploaded:

        if not model:
            st.error("Model file 'Five_Class_Model.h5' not found.")

        else:

            tmp = "temp_single"

            if os.path.exists(tmp):
                shutil.rmtree(tmp)

            os.makedirs(tmp)

            with zipfile.ZipFile(uploaded, "r") as z:
                z.extractall(tmp)

            hea_files = [f for f in os.listdir(tmp) if f.endswith(".hea")]

            if not hea_files:
                st.error("No .hea file found in ZIP.")

            else:

                selected_file = st.selectbox("Select Patient Record", hea_files)

                header_path = os.path.join(tmp, selected_file)

                try:
                    with open(header_path, "r") as f:
                        header_text = f.read()
                except:
                    header_text = ""

                age, sex = extract_metadata(header_text)

                st.subheader("Patient Data")

                st.write(
                    f"Age: {age} | Sex: {'Female' if sex==1 else 'Male'}"
                )

                record_path = os.path.join(
                    tmp, selected_file.replace(".hea", "")
                )

                sig = process_signal(record_path)

                if sig is None:
                    st.error("ECG signal could not be processed.")

                else:

                    clin = np.array([[age/100.0, sex]], dtype=np.float32)

                    try:

                        probs = model.predict([sig, clin], verbose=0)[0]

                        idx = int(np.argmax(probs))

                        dx = CLASS_NAMES[idx]

                        conf = float(probs[idx])

                    except Exception as e:

                        st.error(f"Inference error: {e}")
                        probs = None

                    if probs is not None:

                        # Risk
                        risk, tone = get_risk_assessment(dx)

                        if tone == "error":
                            st.error(risk)
                        elif tone == "warning":
                            st.warning(risk)
                        elif tone == "success":
                            st.success(risk)
                        else:
                            st.info(risk)

                        # Diagnosis metrics
                        c1, c2 = st.columns(2)

                        c1.metric("Primary Diagnosis", dx)

                        c2.metric("AI Confidence", f"{conf*100:.1f}%")

                        # Class Probabilities
                        st.subheader("Class Probabilities")

                        df = pd.DataFrame({
                            "Class": CLASS_NAMES,
                            "Probability": probs
                        })

                        df = df.sort_values(
                            "Probability",
                            ascending=False
                        )

                        st.dataframe(
                            df.style.format(
                                {"Probability": "{:.2%}"}
                            ),
                            use_container_width=True
                        )

                        # ECG GRAPH
                        st.subheader("ECG Signal (Lead I)")

                        fig, ax = plt.subplots(figsize=(10,3))

                        ax.plot(sig[0, :800, 0], linewidth=1)

                        ax.set_xlabel("Time")

                        ax.set_ylabel("Amplitude")

                        ax.grid(True, linestyle="--", alpha=0.3)

                        st.pyplot(fig)
# -----------------------------------
# BULK PREDICTION
# -----------------------------------
elif st.session_state.page == "Bulk Prediction":

    st.header("Bulk Prediction")

    up = st.file_uploader("Upload dataset ZIP", type="zip")

    if up and model:

        if st.button("Start Batch Analysis"):

            tmp = "temp_batch"

            if os.path.exists(tmp):
                shutil.rmtree(tmp)

            os.makedirs(tmp)

            with zipfile.ZipFile(up, "r") as z:
                z.extractall(tmp)

            heas = []

            for r, _, files in os.walk(tmp):
                for f in files:
                    if f.endswith(".hea"):
                        heas.append(os.path.join(r, f))

            results = []
            bar = st.progress(0)

            for i, hp in enumerate(heas):

                content = open(hp).read()

                age, sex = extract_metadata(content)

                sig = process_signal(hp.replace(".hea", ""))

                if sig is not None:

                    clin = np.array([[age/100.0, sex]])

                    probs = model.predict([sig, clin])[0]

                    idx = np.argmax(probs)

                    dx = CLASS_NAMES[idx]

                    conf = probs[idx]

                    risk, _ = get_risk_assessment(dx)

                    results.append({
                        "Patient ID": os.path.basename(hp).split(".")[0],
                        "Age": age,
                        "Sex": "Female" if sex else "Male",
                        "Diagnosis": dx,
                        "Risk Level": risk,
                        "Confidence": f"{conf*100:.1f}%"
                    })

                bar.progress((i+1)/len(heas))

            df = pd.DataFrame(results)

            # STORE RESULTS IN SESSION STATE
            st.session_state["batch_df"] = df
            st.session_state["filter"] = "ALL"

    # -------- DISPLAY SECTION --------

    if "batch_df" in st.session_state:

        df = st.session_state["batch_df"]

        high = df[df["Risk Level"].str.contains("HIGH")]
        moderate = df[df["Risk Level"].str.contains("MODERATE")]
        low = df[df["Risk Level"].str.contains("LOW")]

        st.subheader("Batch Summary")

        c1, c2, c3, c4 = st.columns(4)

        if c1.button(f"Total ({len(df)})"):
            st.session_state["filter"] = "ALL"

        if c2.button(f"High ({len(high)})"):
            st.session_state["filter"] = "HIGH"

        if c3.button(f"Moderate ({len(moderate)})"):
            st.session_state["filter"] = "MODERATE"

        if c4.button(f"Low ({len(low)})"):
            st.session_state["filter"] = "LOW"

        choice = st.session_state.get("filter", "ALL")

        if choice == "HIGH":
            st.dataframe(high, use_container_width=True)

        elif choice == "MODERATE":
            st.dataframe(moderate, use_container_width=True)

        elif choice == "LOW":
            st.dataframe(low, use_container_width=True)

        else:
            st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            "Cardio_Batch_Report.csv"
        )

# -----------------------------------
# PERFORMANCE
# -----------------------------------
elif st.session_state.page == "Performance":

    st.header("Performance Metrics")

    st.subheader("Confusion Matrices")

    images = []

    for f in os.listdir("."):
        if f.endswith(".png") and "Matrix" in f:
            if "Matrix_.png" in f:
                continue
            images.append(f)

    if images:

        cols = st.columns(2)

        for i,img in enumerate(sorted(images)):

            caption = img.replace("Matrix_","").replace(".png","")

            with cols[i % 2]:
                st.image(img, caption=caption)

    else:
        st.warning("No confusion matrices found")

# -----------------------------------
# FOOTER
# -----------------------------------
st.caption("System Status: Online" if model else "System Status: Model Missing")
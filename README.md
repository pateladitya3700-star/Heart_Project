# 🫀 Cardiovascular Disease Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)

> **🚀 [LIVE DEMO - Click Here to Try the App!](https://your-app-url-here.streamlit.app)**

A deep learning-powered web application for predicting cardiovascular diseases using ECG signals. Built with TensorFlow and Streamlit.

---

## 🎯 Features

- **Single Patient Prediction**: Upload individual ECG records for instant diagnosis
- **Bulk Analysis**: Process multiple patient records simultaneously
- **AI-Powered**: Deep learning model with 5-class classification
- **Risk Assessment**: Automatic risk level categorization (High/Moderate/Low)
- **Interactive Dashboard**: Real-time visualization and ranking system
- **Export Results**: Download analysis reports as CSV

---

## 🏥 Supported Diagnoses

The system can detect 5 cardiac conditions:

1. ✅ **Normal Sinus Rhythm** - Healthy heart rhythm
2. ⚠️ **Arrhythmia** - Irregular heart rhythm
3. 🔴 **Atrial Fibrillation** - Rapid, irregular atrial rhythm
4. 🔴 **Myocardial Infarction** - Heart attack
5. 🔴 **ST Depression** - Indicator of myocardial ischemia

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.13
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pateladitya3700-star/Heart_Project.git
cd Heart_Project
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 How It Works

1. **Upload ECG Data**: Provide patient ECG records (.zip with .hea + .dat files)
2. **AI Analysis**: Deep learning model processes the ECG signals
3. **Get Results**: View diagnosis, confidence score, and risk level
4. **Export Report**: Download results for medical records

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.13
- **ML Framework**: TensorFlow 2.21
- **Data Processing**: NumPy, Pandas
- **Signal Processing**: WFDB
- **Visualization**: Matplotlib

---

## 📁 Project Structure

```
Heart_Project/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Five_Class_Model.h5   # Trained deep learning model
├── background.jpg         # UI background image
├── Matrix_*.png          # Performance metrics images
└── README.md             # Project documentation
```

---

## 🎓 Model Performance

The deep learning model achieves high accuracy across all 5 cardiac conditions. Performance metrics and confusion matrices are available in the app's Performance section.

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Aditya Patel**
- GitHub: [@pateladitya3700-star](https://github.com/pateladitya3700-star)

---

## 🙏 Acknowledgments

- PTB-XL ECG Database for training data
- TensorFlow and Streamlit communities
- Medical professionals for domain expertise

---

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

---

**⭐ If you find this project useful, please consider giving it a star!**

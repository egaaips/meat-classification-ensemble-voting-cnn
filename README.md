# 🎈 Blank app template
# Klasifikasi Jenis Daging Menggunakan CNN dan Hard Voting Ensemble

Proyek ini bertujuan untuk mengklasifikasikan gambar daging menjadi tiga kategori: daging sapi, kambing, dan babi. Model ini dikembangkan menggunakan arsitektur Convolutional Neural Network (CNN) seperti ResNet50V2, InceptionV3, dan MobileNetV2, serta dikombinasikan dengan metode ensemble learning *hard voting* untuk meningkatkan akurasi klasifikasi.

## 📁 Struktur Folder

/project-folder │ 
├── data_split/
│ └── labels.npy 
├── datasets/
│ └── daging babi 
│ └── daging kambing
│ └── daging babi
├── model/ 
│ └── model1_ResNet50V2.keras 
│ └── model2_InceptionV3.keras 
│ └── model3_MobileNetV2.keras 
├── streamlit_app.py 
└── README.md

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

Catatan: Jika model tidak tersedia karena ukurannya melebihi batas GitHub, silakan unduh melalui tautan berikut:
📎 [Models-Pretrained](https://drive.google.com/drive/folders/1l6jF-sHaHt7nU6WH4a_ztiJTiJGYzTf-?usp=sharing)
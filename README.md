# 📊 Machine Learning System - Insurance Cost Prediction

Proyek ini merupakan implementasi sistem machine learning end-to-end untuk prediksi biaya asuransi, mencakup:

* Data preprocessing
* Model training dengan MLflow
* Hyperparameter tuning
* Monitoring (Prometheus & Grafana)

---

# ⚙️ 1. Setup Environment

Gunakan Python versi:

```bash
Python 3.12.x
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📁 2. Struktur Project

```
submission/
├── Eksperimen_SML_Ayudha/
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── insurance_preprocessing/
│   │   ├── insurance_train_preprocessed.csv
│   │   ├── insurance_test_preprocessed.csv
│   │   └── columns.csv
│   ├── requirements.txt
│   ├── screenshot_dashboard.jpg
│   └── screenshot_artifak.jpg
├── Monitoring dan Logging/
│   ├── inference.py
│   ├── prometheus_exporter.py
│   └── prometheus.yml
```

---

# 🔄 3. Menjalankan Preprocessing

Masuk ke folder eksperimen:

```bash
cd Eksperimen_SML_Ayudha
```

Jalankan:

```bash
python run_preprocessing.py
```

Output:

```
preprocessing/insurance_preprocessing/
├── insurance_train_preprocessed.csv
├── insurance_test_preprocessed.csv
└── columns.csv
```

---

# 🤖 4. Menjalankan Model (MLflow)

## Jalankan MLflow UI

```bash
mlflow ui
```

Akses di browser:

```
http://127.0.0.1:5000
```

---

## Jalankan Training Basic

```bash
cd ../Membangun_model
python modelling.py
```

---

## Jalankan Hyperparameter Tuning

```bash
python modelling_tuning.py
```

Output:

* Metrics akan tampil di terminal
* Logging tersimpan di MLflow UI

---

# 📊 5. Monitoring (Prometheus & Grafana)

## Jalankan Prometheus Exporter

```bash
cd ../Monitoring dan Logging
python prometheus_exporter.py
```

---

## Jalankan Prometheus

```bash
prometheus --config.file=prometheus.yml
```

Akses:

```
http://localhost:9090
```

---

## Grafana

1. Jalankan Grafana
2. Tambahkan data source → Prometheus
3. Import dashboard monitoring

---

# 🚀 6. Inference

Jalankan file inference:

```bash
python inference.py
```

Output:

```
Prediksi biaya asuransi
```

---

# 📌 7. Catatan

* Pastikan path file sesuai dengan struktur folder
* Pastikan MLflow UI berjalan sebelum training
* Gunakan environment yang sama untuk semua proses

---

# 👨‍💻 Author

Nama: Ayudha
Project: Machine Learning System (Dicoding Submission)


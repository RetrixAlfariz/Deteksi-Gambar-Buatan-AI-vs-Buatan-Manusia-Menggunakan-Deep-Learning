# Deteksi Gambar AI vs Manusia Menggunakan Deep Learning

## 📌 Gambaran Umum Proyek
Proyek ini berfokus pada pembangunan **model klasifikasi gambar berbasis deep learning** untuk membedakan antara **gambar hasil buatan AI** dan **gambar hasil buatan manusia**.  
Seiring berkembangnya model generatif AI, gambar sintetis menjadi semakin realistis dan semakin sulit dibedakan dari gambar asli. Hal ini menimbulkan tantangan penting dalam bidang seperti **keaslian media**, **forensik digital**, dan **moderasi konten**.

Untuk menjawab permasalahan tersebut, proyek ini mengembangkan arsitektur **Convolutional Neural Network (CNN)** kustom yang diperkuat dengan **Residual Connection** dan **Squeeze-and-Excitation (SE) Block**, kemudian mengintegrasikan model terlatih ke dalam sebuah **prototype web app lokal** untuk melakukan prediksi gambar secara langsung.

---

## 🎯 Tujuan Proyek
Tujuan utama dari proyek ini adalah:

- Membangun model deep learning untuk mengklasifikasikan apakah sebuah gambar termasuk:
  - **AI-generated**
  - **Human-created**
- Melakukan analisis awal dataset melalui **Exploratory Data Analysis (EDA)**
- Menerapkan teknik **preprocessing** dan **augmentasi gambar**
- Melatih dan mengevaluasi model klasifikasi berbasis CNN
- Mengembangkan **prototype web app lokal** untuk prediksi gambar secara real-time

---

## 🗂️ Dataset
Dataset yang digunakan dalam proyek ini adalah:

- **Nama:** ShutterStock AI vs. Human-Generated Image Dataset
- **Sumber:** Kaggle
- **Link:** https://www.kaggle.com/datasets/shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image
- **Total Gambar:** 100.000
- **Data Latih:** 80.000 gambar
- **Data Uji:** 20.000 gambar
- **Kelas:**
  - `0` = Human-created
  - `1` = AI-generated

### Karakteristik Dataset
- Dataset terdiri dari campuran:
  - foto asli
  - ilustrasi buatan manusia
  - gambar sintetis hasil AI
- Cocok untuk tugas **klasifikasi citra biner**
- Relevan untuk penelitian dalam bidang:
  - deteksi gambar hasil AI
  - computer vision
  - forensik digital
  - verifikasi keaslian media

---

## 📊 Exploratory Data Analysis (EDA)

Analisis eksploratif dasar dilakukan untuk memahami karakteristik awal dataset sebelum proses pelatihan model.

### EDA yang Dilakukan
- Visualisasi **distribusi label**
- Visualisasi **contoh gambar** dari kedua kelas:
  - Human-created images
  - AI-generated images

### Temuan Utama
- Dataset latih berada dalam kondisi **seimbang**
- Secara visual, gambar hasil AI dan gambar buatan manusia tampak **cukup mirip**
- Hal ini menunjukkan bahwa membedakan kedua kelas tersebut cukup menantang jika hanya mengandalkan pengamatan manusia

---

## 🧹 Preprocessing

Sebelum model dilatih, dilakukan tahap preprocessing untuk memastikan format input seragam dan membantu meningkatkan kemampuan generalisasi model.

### Tahapan Preprocessing
- Mengubah seluruh gambar ke format **RGB**
- Mengubah ukuran seluruh gambar menjadi **224 × 224**
- Melakukan normalisasi nilai piksel menggunakan normalisasi bergaya ImageNet:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

### Data Augmentation (Khusus Data Latih)
Beberapa teknik augmentasi yang digunakan antara lain:

- `RandomHorizontalFlip`
- `RandomVerticalFlip (p=0.2)`
- `RandomRotation (15°)`
- `ColorJitter`
- `RandomGrayscale (p=0.05)`

### Pembagian Data
Data latih kemudian dibagi kembali menjadi:

- **Training Set:** 85%
- **Validation Set:** 15%

Pembagian dilakukan menggunakan **stratified split** agar distribusi label tetap seimbang pada masing-masing subset.

---

## 🧠 Arsitektur Model

Proyek ini menggunakan arsitektur **CNN kustom** yang dirancang untuk tugas klasifikasi citra biner.

### Komponen Utama
- **Convolutional Neural Network (CNN)**
- **Residual Convolutional Blocks**
- **Batch Normalization**
- **ReLU Activation**
- **Squeeze-and-Excitation (SE) Blocks**
- **Adaptive Global Average Pooling**
- **Fully Connected Classification Head**
- **Dropout Regularization**

### Karakteristik Arsitektur
- Ukuran input gambar: **224 × 224**
- Ekstraksi fitur dilakukan secara bertahap dari pola visual tingkat rendah hingga tingkat tinggi
- Residual connection membantu optimasi pada jaringan yang lebih dalam
- SE Block membantu model memfokuskan perhatian pada channel fitur yang paling relevan

### Alasan Pemilihan Arsitektur
Perbedaan antara gambar hasil AI dan gambar buatan manusia sering kali bersifat halus dan sulit dibedakan secara visual.  
Oleh karena itu, digunakan arsitektur CNN yang lebih mendalam dengan residual learning dan channel attention agar model mampu mengekstraksi pola visual yang lebih bermakna.

---

## ⚙️ Strategi Pelatihan Model

Model dilatih menggunakan strategi pelatihan yang dirancang untuk meningkatkan performa sekaligus mengurangi risiko overfitting.

### Konfigurasi Pelatihan
- **Maksimum Epoch:** 30
- **Batch Size:** 32

### Fungsi Loss
- `BCEWithLogitsLoss`

Fungsi ini digunakan karena permasalahan yang dihadapi merupakan **klasifikasi biner**, dan model menghasilkan satu nilai output dalam bentuk **logit**.

### Optimizer
- `AdamW`
- Learning Rate: `0.001`
- Weight Decay: `0.0001`

### Learning Rate Scheduler
- `CosineAnnealingLR`
- `T_max = 30`
- `eta_min = 1e-6`

### Early Stopping
Untuk mencegah overfitting, diterapkan **Early Stopping** dengan konfigurasi:

- **Patience:** 5
- **Min Delta:** 0.0005
- **Metrik yang Dipantau:** Validation AUC-ROC

### Model Checkpoint
Model dengan performa terbaik selama pelatihan akan disimpan secara otomatis.

---

## 📈 Evaluasi Model

Performa model dievaluasi menggunakan beberapa metrik klasifikasi.

### Metrik Evaluasi
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### Performa Terbaik pada Validation Set
- **Validation Accuracy:** ~99%
- **Validation AUC-ROC:** **0.9995**

### Classification Report
| Kelas | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Human | 1.00 | 0.99 | 0.99 |
| AI    | 0.99 | 1.00 | 0.99 |

### Interpretasi Hasil
Model menunjukkan performa yang sangat baik pada data validasi, yang mengindikasikan bahwa model mampu mempelajari pola visual yang relevan untuk membedakan gambar hasil AI dan gambar buatan manusia.

Namun demikian, hasil ini tetap perlu diinterpretasikan secara hati-hati, terutama jika model ingin diterapkan pada kondisi dunia nyata.

---

## 🌍 Keterbatasan pada Dunia Nyata

Meskipun model memperoleh performa yang sangat tinggi pada dataset yang digunakan, **performa di dunia nyata bisa berbeda**.

### Hal-Hal yang Perlu Diperhatikan
- Gambar hasil AI di dunia nyata terus berkembang menjadi **semakin realistis**
- Model generatif terbaru dapat menghasilkan gambar yang **lebih sulit dideteksi**
- Gambar dunia nyata dapat memiliki:
  - artefak kompresi yang berbeda
  - jejak editing tambahan
  - gaya visual yang tidak ada di dataset pelatihan
  - distribusi visual yang belum pernah dilihat model

### Insight Penting
Model ini bekerja sangat baik pada dataset penelitian yang digunakan, tetapi **belum tentu langsung memiliki kemampuan generalisasi yang kuat** terhadap semua gambar AI yang beredar di dunia nyata.

Hal ini penting karena:
> model generatif AI modern berkembang sangat cepat, sehingga perbedaan visual antara gambar sintetis dan gambar buatan manusia semakin mengecil.

---

# Prototype Web Application
![alt text](artifacts/web-preview-current.png)
=======
# Deteksi Gambar AI vs Manusia Menggunakan Deep Learning

## 📌 Gambaran Umum Proyek
Proyek ini berfokus pada pembangunan **model klasifikasi gambar berbasis deep learning** untuk membedakan antara **gambar hasil buatan AI** dan **gambar hasil buatan manusia**.  
Seiring berkembangnya model generatif AI, gambar sintetis menjadi semakin realistis dan semakin sulit dibedakan dari gambar asli. Hal ini menimbulkan tantangan penting dalam bidang seperti **keaslian media**, **forensik digital**, dan **moderasi konten**.

Untuk menjawab permasalahan tersebut, proyek ini mengembangkan arsitektur **Convolutional Neural Network (CNN)** kustom yang diperkuat dengan **Residual Connection** dan **Squeeze-and-Excitation (SE) Block**, kemudian mengintegrasikan model terlatih ke dalam sebuah **prototype web app lokal** untuk melakukan prediksi gambar secara langsung.

---

## 🎯 Tujuan Proyek
Tujuan utama dari proyek ini adalah:

- Membangun model deep learning untuk mengklasifikasikan apakah sebuah gambar termasuk:
  - **AI-generated**
  - **Human-created**
- Melakukan analisis awal dataset melalui **Exploratory Data Analysis (EDA)**
- Menerapkan teknik **preprocessing** dan **augmentasi gambar**
- Melatih dan mengevaluasi model klasifikasi berbasis CNN
- Mengembangkan **prototype web app lokal** untuk prediksi gambar secara real-time

---

## 🗂️ Dataset
Dataset yang digunakan dalam proyek ini adalah:

- **Nama:** ShutterStock AI vs. Human-Generated Image Dataset
- **Sumber:** Kaggle
- **Link:** https://www.kaggle.com/datasets/shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image
- **Total Gambar:** 100.000
- **Data Latih:** 80.000 gambar
- **Data Uji:** 20.000 gambar
- **Kelas:**
  - `0` = Human-created
  - `1` = AI-generated

### Karakteristik Dataset
- Dataset terdiri dari campuran:
  - foto asli
  - ilustrasi buatan manusia
  - gambar sintetis hasil AI
- Cocok untuk tugas **klasifikasi citra biner**
- Relevan untuk penelitian dalam bidang:
  - deteksi gambar hasil AI
  - computer vision
  - forensik digital
  - verifikasi keaslian media

---

## 📊 Exploratory Data Analysis (EDA)

Analisis eksploratif dasar dilakukan untuk memahami karakteristik awal dataset sebelum proses pelatihan model.

### EDA yang Dilakukan
- Visualisasi **distribusi label**
- Visualisasi **contoh gambar** dari kedua kelas:
  - Human-created images
  - AI-generated images

### Temuan Utama
- Dataset latih berada dalam kondisi **seimbang**
- Secara visual, gambar hasil AI dan gambar buatan manusia tampak **cukup mirip**
- Hal ini menunjukkan bahwa membedakan kedua kelas tersebut cukup menantang jika hanya mengandalkan pengamatan manusia

---

## 🧹 Preprocessing

Sebelum model dilatih, dilakukan tahap preprocessing untuk memastikan format input seragam dan membantu meningkatkan kemampuan generalisasi model.

### Tahapan Preprocessing
- Mengubah seluruh gambar ke format **RGB**
- Mengubah ukuran seluruh gambar menjadi **224 × 224**
- Melakukan normalisasi nilai piksel menggunakan normalisasi bergaya ImageNet:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

### Data Augmentation (Khusus Data Latih)
Beberapa teknik augmentasi yang digunakan antara lain:

- `RandomHorizontalFlip`
- `RandomVerticalFlip (p=0.2)`
- `RandomRotation (15°)`
- `ColorJitter`
- `RandomGrayscale (p=0.05)`

### Pembagian Data
Data latih kemudian dibagi kembali menjadi:

- **Training Set:** 85%
- **Validation Set:** 15%

Pembagian dilakukan menggunakan **stratified split** agar distribusi label tetap seimbang pada masing-masing subset.

---

## 🧠 Arsitektur Model

Proyek ini menggunakan arsitektur **CNN kustom** yang dirancang untuk tugas klasifikasi citra biner.

### Komponen Utama
- **Convolutional Neural Network (CNN)**
- **Residual Convolutional Blocks**
- **Batch Normalization**
- **ReLU Activation**
- **Squeeze-and-Excitation (SE) Blocks**
- **Adaptive Global Average Pooling**
- **Fully Connected Classification Head**
- **Dropout Regularization**

### Karakteristik Arsitektur
- Ukuran input gambar: **224 × 224**
- Ekstraksi fitur dilakukan secara bertahap dari pola visual tingkat rendah hingga tingkat tinggi
- Residual connection membantu optimasi pada jaringan yang lebih dalam
- SE Block membantu model memfokuskan perhatian pada channel fitur yang paling relevan

### Alasan Pemilihan Arsitektur
Perbedaan antara gambar hasil AI dan gambar buatan manusia sering kali bersifat halus dan sulit dibedakan secara visual.  
Oleh karena itu, digunakan arsitektur CNN yang lebih mendalam dengan residual learning dan channel attention agar model mampu mengekstraksi pola visual yang lebih bermakna.

---

## ⚙️ Strategi Pelatihan Model

Model dilatih menggunakan strategi pelatihan yang dirancang untuk meningkatkan performa sekaligus mengurangi risiko overfitting.

### Konfigurasi Pelatihan
- **Maksimum Epoch:** 30
- **Batch Size:** 32

### Fungsi Loss
- `BCEWithLogitsLoss`

Fungsi ini digunakan karena permasalahan yang dihadapi merupakan **klasifikasi biner**, dan model menghasilkan satu nilai output dalam bentuk **logit**.

### Optimizer
- `AdamW`
- Learning Rate: `0.001`
- Weight Decay: `0.0001`

### Learning Rate Scheduler
- `CosineAnnealingLR`
- `T_max = 30`
- `eta_min = 1e-6`

### Early Stopping
Untuk mencegah overfitting, diterapkan **Early Stopping** dengan konfigurasi:

- **Patience:** 5
- **Min Delta:** 0.0005
- **Metrik yang Dipantau:** Validation AUC-ROC

### Model Checkpoint
Model dengan performa terbaik selama pelatihan akan disimpan secara otomatis.

---

## 📈 Evaluasi Model

Performa model dievaluasi menggunakan beberapa metrik klasifikasi.

### Metrik Evaluasi
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### Performa Terbaik pada Validation Set
- **Validation Accuracy:** ~99%
- **Validation AUC-ROC:** **0.9995**

### Classification Report
| Kelas | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Human | 1.00 | 0.99 | 0.99 |
| AI    | 0.99 | 1.00 | 0.99 |

### Interpretasi Hasil
Model menunjukkan performa yang sangat baik pada data validasi, yang mengindikasikan bahwa model mampu mempelajari pola visual yang relevan untuk membedakan gambar hasil AI dan gambar buatan manusia.

Namun demikian, hasil ini tetap perlu diinterpretasikan secara hati-hati, terutama jika model ingin diterapkan pada kondisi dunia nyata.

---

## 🌍 Keterbatasan pada Dunia Nyata

Meskipun model memperoleh performa yang sangat tinggi pada dataset yang digunakan, **performa di dunia nyata bisa berbeda**.

### Hal-Hal yang Perlu Diperhatikan
- Gambar hasil AI di dunia nyata terus berkembang menjadi **semakin realistis**
- Model generatif terbaru dapat menghasilkan gambar yang **lebih sulit dideteksi**
- Gambar dunia nyata dapat memiliki:
  - artefak kompresi yang berbeda
  - jejak editing tambahan
  - gaya visual yang tidak ada di dataset pelatihan
  - distribusi visual yang belum pernah dilihat model

### Insight Penting
Model ini bekerja sangat baik pada dataset penelitian yang digunakan, tetapi **belum tentu langsung memiliki kemampuan generalisasi yang kuat** terhadap semua gambar AI yang beredar di dunia nyata.

Hal ini penting karena:
> model generatif AI modern berkembang sangat cepat, sehingga perbedaan visual antara gambar sintetis dan gambar buatan manusia semakin mengecil.

---

# Prototype Web Application
![alt text](artifacts/web-preview-current.png)

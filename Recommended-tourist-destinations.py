# -*- coding: utf-8 -*-
"""ML_EXPERT_SUBMISSION_Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pnXkeSj7PRSvs5dKVY6TCFSQSZ43nPve

# Submission Final : Recommendation System
____________________________________________________________
____________________________________________________________

# Data Diri
____________________________________________________________
____________________________________________________________

Nama            : **Doli sawaluddin**

E-mail Dicoding : **dholys7@gmail.com**
____________________________________________________________
____________________________________________________________

Dataset:
*   [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

References:


*   [Sistem rekomendasi- Content Based](https://mti.binus.ac.id/2020/11/17/sistem-rekomendasi-content-based/)
*   [Memahami Data Dengan Exploratory Data Analysis](https://medium.com/data-folks-indonesia/memahami-data-dengan-exploratory-data-analysis-a53b230cce84)
*   [Membuat Sistem Rekomendasi dengan Python — Part 1 Teori dan Penjelasan | Machine Learning](https://rizki4106.medium.com/membuat-sistem-rekomendasi-dengan-python-part-1-teori-dan-penjelasan-machine-learning-a567afe5b7f8)

# 1. Import Library yang diperlukan
"""

import os
import zipfile
from google.colab import files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""**Import Library yang diperlukan:**

- **`os`:** Merupakan Sebuah Library yang digunakan untuk berinteraksi dengan sistem operasi, seperti mengakses file dan direktori.
- **`zipfile`:** Merupakan Sebuah Library yang digunakan  untuk mengolah file zip, seperti mengekstrak atau membuat file zip.
- **`google.colab.files`:** Merupakan Sebuah Library yang digunakan untuk berinteraksi dengan file di Google Colab, seperti mengunggah atau mengunduh file.
- **`pandas`:** adalah Library untuk mengolah data tabular, seperti membaca, menulis, dan memanipulasi DataFrame.
- **`numpy`:** Library ini berguna untuk komputasi numerik, seperti array dan matriks.
- **`matplotlib.pyplot`:** Library ini berfungsi untuk membuat plot dan grafik.
- **`seaborn`:** Library untuk membuat visualisasi data statistik yang lebih baik dan canggih.
- **`Library sklearn`** untuk melakukan pemrosesan machine learning dan data analysis.


* **`tensorflow as tf`:**  Kode ini mengimpor library TensorFlow dan memberikan alias `tf` untuk mempermudah penggunaan.
* **`from tensorflow import keras`:**  Memperkenalkan Keras, API tingkat tinggi yang dibangun di atas TensorFlow. Keras menyediakan antarmuka yang mudah digunakan untuk membangun dan melatih model deep learning.
* **`from tensorflow.keras import layers`:** Mengimpor modul `layers` dari Keras. Modul ini berisi berbagai jenis layer yang digunakan untuk membangun arsitektur neural network, seperti lapisan Dense (fully connected), Convolutional, dan lainnya.
* **`from tensorflow.keras.optimizers import Adam`:** Mengimpor optimiser Adam dari Keras. Optimiser digunakan untuk memperbarui bobot model selama proses pelatihan. Adam merupakan optimiser yang populer karena efektivitas dan efisiensi dalam banyak kasus.
* **`from tensorflow.keras.losses import BinaryCrossentropy`:** Mengimpor fungsi kerugian Binary Crossentropy dari Keras. Fungsi kerugian digunakan untuk mengukur kesalahan model selama pelatihan, dan fungsi ini sangat cocok untuk masalah klasifikasi biner.
* **`from tensorflow.keras.metrics import RootMeanSquaredError`:** Mengimpor metrik Root Mean Squared Error (RMSE) dari Keras. Metrik digunakan untuk mengevaluasi kinerja model selama pelatihan dan pengujian.
* **`from tensorflow.keras.callbacks import EarlyStopping`:** Mengimpor callback `EarlyStopping` dari Keras. Callback ini digunakan untuk menghentikan proses pelatihan model lebih awal jika kinerja model tidak membaik dalam beberapa epoch.



**Catatan:**
- Library-library tersebut digunakan dalam konteks proyek sistem rekomendasi
- Import library dilakukan di awal script untuk memastikan bahwa library-library yang dibutuhkan tersedia sebelum script dijalankan.

# 2. Data Loading

## 2.1. Download Dataset

**Mengunduh dataset dari kaggle**
"""

files.upload()

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets list -s "indonesia-tourism-destination"
!kaggle datasets download -d aprabowo/indonesia-tourism-destination

"""## 2.2. Dataset Preparation

Melakukan Ekstraksi pada file yang telah didownload, dan menampilkan isi dari dataset  kedalam dataframe dengan memanfaatkan *library pandas.*
"""

local_zip = '/content/indonesia-tourism-destination.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

ratings_df = pd.read_csv('tourism_rating.csv')
user_df = pd.read_csv('user.csv')
tourismid_df = pd.read_csv('tourism_with_id.csv')

ratings_df

user_df

tourismid_df

"""**Mengecek Nilai yang hilang dan nilai kolom yang duplikat**

Jika kita perhatikan kolom unnamed:11 dan unnamed:12 terdapat missing value pada kolom unnamed:11 dan nilai yang sama antara 	Place_Id dengan unnamed:12.

**Nilai yang hilang**
"""

print(tourismid_df[['Unnamed: 11', 'Unnamed: 12']])
nan_count = tourismid_df['Unnamed: 11'].isnull().sum()
print()
print("Jumlah NaN pada kolom Unnamed: 11:", nan_count)

"""**Nilai yang sama**"""

print(tourismid_df[['Place_Id', 'Unnamed: 12']])

"""Jika kita perhatikan terdapat missing value pada kolom unnamed:11 sebanyak 437 data dan nilai yang sama antara 	Place_Id dengan unnamed:12. Kita dapat mengghapus kedua kolom ini agar data menjadi bersih. Kita juga akan menghapus time_minutes, cordinate, lat, dan long karena data lokasi dan waktu tidak diperlukan untuk tahapan selanjutnya.

Untuk rating kita akan menggunakan rating dari dataframe `ratings_df` agar lebih sesuai dengan preferensi pengguna, sehingga kita akan menghapus data rating pada `tourismid_df` karena untuk sistem rekomendasi ini tidak akan dipakai.

**Menghapus kolom yang tidak diperlukan**
"""

tourismid_df = tourismid_df.drop(['Unnamed: 11', 'Unnamed: 12','Time_Minutes','Coordinate','Lat','Long','Rating'], axis=1)
tourismid_df

"""Berdasarkan data diatas untuk membuat sistem rekomendasi , kita hanya akan memanfaatkan dataframe :


*   `user_df` : Berisi informasi tentang user
*   `ratings_df` : Berisi Informasi rating yang diberikan user
*   `tourismid_df` : Berisi informasi lokasi wisata

Alasan kenapa kita tidak menggunakan `dataset package `adalah karena kita belum mengetahui package mana yang pernah diambil oleh user, sehingga kita hanya bisa mencocokkan kesesuaian user dan package hanya berdasarkan kota yang dia kunjungi.

# 3. Data Understanding

## 3.1 Jumlah Data Masing-masing Atribut dari Dataset

Melihat jumlah data atribut penting yang ada pada masing-masing dataframe dengan menggunakan fungsi `.unique().`
"""

print('Jumlah data Pengguna:', len(user_df.User_Id.unique()))
print('Jumlah data Destinasi Wisata:', len(tourismid_df.Place_Id.unique()))
print('Jumlah Kota', len(tourismid_df.City.unique()))
print('Jumlah user yang memberikan Rating:', len(ratings_df.User_Id.unique()))
print('Jumlah data Rating:', len(ratings_df.User_Id))

"""Dari data diatas kita mendapatkan 300 pengguna, 437 destinasi wisata dalam 5 kota dengan jumlah rating mencapai 10.000

## 3.2 Univariate Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) adalah bagian dari proses data science. EDA menjadi sangat penting sebelum melakukan feature engineering dan modeling karena dalam tahap ini kita harus memahami datanya terlebih dahulu.Exploratory Data Analysis memungkinkan analyst memahami isi data yang digunakan, mulai dari distribusi, frekuensi, korelasi dan lainnya [[1]](https://medium.com/data-folks-indonesia/memahami-data-dengan-exploratory-data-analysis-a53b230cce84).

### 3.2.1 Dataset user

Exploratory Data Analysis (EDA) untuk dataframe *user_df*.
"""

user_df

"""Terdapat 300 pengguna dari berbagai wilayah dan umur yang berbeda.



"""

user_df.info()

"""Terdapat 2 variabel bertype int64 dan 1 variabel bertype object yang merupakan alamat user."""

user_df.describe().astype('int')

"""Berdasarkan data diatas terdapat 300 pengguna dengan detail sebagai berikut:

*   **count:** Jumlah total data yang dianalisis. Dalam hal ini, ada 300 data untuk kedua variabel, yang berarti ada 300 pengguna dalam dataset ini.
*  **mean:** Rata-rata atau nilai tengah dari semua data. Rata-rata usia pengguna adalah 28 tahun.
*   **std:** Standar deviasi, yang merupakan ukuran sebaran data dari rata-rata. Semakin kecil nilai standar deviasi, semakin dekat data ke nilai rata-rata. Dalam hal ini, standar deviasi usia adalah 6, yang berarti usia pengguna cenderung tersebar dalam rentang 6 tahun di sekitar rata-rata 28 tahun.
*   **min:** Nilai minimum. Usia pengguna termuda adalah 18 tahun.
*  **25%:** Kuartil pertama. Artinya, 25% dari pengguna berusia 24 tahun atau lebih muda.
*  **50%:** Median atau kuartil kedua. Ini adalah nilai tengah dari data yang telah diurutkan. Jadi, 50% pengguna berusia 29 tahun atau lebih muda.
*   **75%:** Kuartil ketiga. Artinya, 75% dari pengguna berusia 34 tahun atau lebih muda
*   **max:** Nilai maksimum. Usia pengguna tertua adalah 40 tahun.

**Menampilkan wisatawan berdasarkan daerah asal**
"""

location_counts = user_df['Location'].value_counts()

plt.figure(figsize=(12, 6))
sns.countplot(x='Location', data=user_df, order=location_counts.index)
plt.xlabel('Location')
plt.ylabel('Number of Users')
plt.title('User Location Distribution')
plt.xticks(rotation=90)
plt.show()

"""Dari data diatas dapat diambil kesimpulan, wisatawan yang memiliki potensi untuk liburan adalah dari daerah Bekasi, Jawa Barat. dan jumlah asal wisatawan dari rentang 10 sampai 22 juga memiliki potensi untuk liburan jika kita bisa merekomendasikan destinasi wisata yang sesuai.

### 3.2.1 Dataset tourism_with_id

Exploratory Data Analysis (EDA) untuk dataframe *tourismid_df*.
"""

tourismid_df

"""Berdasarkan data diatas, kita mendapatkan 437 destinasi wisata

**Menampilkan jumlah kategori wisata**
"""

category_counts = tourismid_df['Category'].value_counts()

print("Category Counts:")
for category, count in category_counts.items():
  print(f"- {category}: {count}")

"""Berdasarkan jumlah kategori wisata diatas, diketahui wisata paling diminati adalah wisata taman hiburan, budaya dan cagar alam."""

tourismid_df.info()

"""Tedapat 4 variabel bertype object yang merupakan nama wisata, deskripsi wisata, kategori wisata, dan kota wisata. Serta terdapat 2 variabel bertype int64 yang merupakan place_id(id lokasi) dan price/tarif wisata."""

tourismid_df.describe().astype('int')

"""Bisa kita lihat biaya atau tarif wisata sangat bervariasi mulai dari 0 rupiah (gratis) sampai 900ribu dalam IDR.

### 3.2.1 Dataset tourism_rating

Exploratory Data Analysis (EDA) untuk dataframe *ratings_df*.
"""

ratings_df

"""Terdapat 3 variabel yaitu **user_id** sebagai pengenal pengguna, **place_id** yang merujuk kealamat wisata dan **place_ratings** yang merupakan rating yang diberikan oleh pengguna.

"""

ratings_df.info()

"""Semua variabel bertype int64"""

ratings_df.describe().astype('int')

"""Ratings yang diberikan pengguna mulai dari 1 sampai yang tertinggi di angka 5. serta jumlah pengguna yang memberikan rating berjumlah 300 orang dengan total rating mencapai 10 ribu.

# 4. Data Preprocessing

Data preprocessing adalah teknik transformasi data mentah menjadi format yang lebih terstruktur dan akurat, sehingga dapat meningkatkan kualitas hasil pemodelan.

## 4.1 Menggabungkan data rating dan data lokasi wisata

Walaupun dengan data `ratings_df` dan `tourismid_id` saja sudah cukup untuk membuat sistem rekomendasi namun untuk memudahkan dalam membaca data, kita akan menggabungkan dataframe `ratings_df` dan `tourismid_df` kedalam variabel `data_wisata`.
"""

data_wisata = pd.merge(ratings_df, tourismid_df, on='Place_Id')
data_wisata

data_wisata.info()

UserAll = np.sort(np.unique(data_wisata.User_Id.unique()))
PlaceAll = np.sort(np.unique(data_wisata.Place_Id.unique()))
print(f'Jumlah User setelah penggabungan  : {len(UserAll)}')
print(f'Jumlah Place setelah penggabungan : {len(PlaceAll)}')

"""Kita mendapatkan 10 ribu data baru dan tidak ada perubahan pada jumlah user maupun place name, selanjutnya kita akan melakukan pengecekan kembali terhadap data yang sudah digabungkan pada data preparation, untuk memastikan apakah ada missing value atau duplikasi pada data.

# 5. Data Preparation

Tahap data preparation berperan penting dalam memastikan bahwa data yang digunakan untuk membangun model adalah data yang berkualitas, relevan, dan siap digunakan pada proses pengembangan model machine learning.

## 5.1 Mengatasi Missing Value

**Mengecek Missing value pada setiap dataframe**
"""

tourismid_df.isnull().sum()

ratings_df.isnull().sum()

user_df.isnull().sum()

data_wisata.isnull().sum()

"""Berdasarkan data diatas ,ternyata tidak ditemukan adanya missing value pada tiap dataframe

## 5.2 Mengatasi duplikasi data
"""

print(f'Jumlah data wisata  yang duplikat: {tourismid_df.duplicated().sum()}')
print(f'Jumlah data rating yang duplikat: {ratings_df.duplicated().sum()}')
print(f'Jumlah data users  yang duplikat: {user_df.duplicated().sum()}')
print(f'Jumlah data Gabungan  yang duplikat: {data_wisata.duplicated().sum()}')

"""Berdasarkan data di atas, dapat dilihat bahwa terdapat data duplikat pada data rating dan data gabungan. dimana masing-masing memiliki 79 data duplikat.

**Memastikan duplikasi pada dataframe**

**ratings_df**
"""

duplicate_rows_df1 = ratings_df[ratings_df.duplicated(keep=False)]

sorted_duplicate_rows_df1 = duplicate_rows_df1.sort_values(by=list(ratings_df.columns))

sorted_duplicate_rows_df1

"""**data_wisata**"""

duplicate_rows_df2 = data_wisata[data_wisata.duplicated(keep=False)]

sorted_duplicate_rows_df2 = duplicate_rows_df2.sort_values(by=list(data_wisata.columns))

sorted_duplicate_rows_df2

"""Berdasarkan data diatas kita mendapatkan 79 data berserta duplikasinya sebanyak 79 data sehingga data yg ditampilkan mencapai 158 baris, dimana bisa kita lihat terdapat beberapa baris yang menampilkan data yang sama. Selanjutnya kita bisa menghapus data duplikat ini.

**Menghapus duplikasi dan Mengecek kembali jumlah duplikat setelah penghapusan**
"""

ratings_df = ratings_df.drop_duplicates()
duplicate_rows_df = ratings_df[ratings_df.duplicated(keep=False)]
print("Jumlah duplikat Rating setelah penghapusan:", len(duplicate_rows_df))

data_wisata = data_wisata.drop_duplicates()
duplicate_rows_df = data_wisata[data_wisata.duplicated(keep=False)]
print("Jumlah duplikat Gabungan setelah penghapusan:", len(duplicate_rows_df))

ratings_df

data_wisata

"""Sekarang data sudah bersih dengan jumlah akhir sebanyak 9921 baris dengan 10 kolom

# 6. Model Development dengan Content Based Filtering

Content-based filtering Memberikan rekomendasi berdasarkan kemiripan atribut dari item atau barang yang disukai oleh pengguna. [[2]](https://mti.binus.ac.id/2020/11/17/sistem-rekomendasi-content-based/)

Dengan sistem rekomendasi berbasis konten, pengguna akan mendapatkan saran yang lebih personal dan sesuai dengan minat mereka.

## 6.1 TF-IDF Vectorizer

TF-IDF digunakan untuk mengubah teks menjadi vektor numerik yang merepresentasikan pentingnya setiap kata dalam dokumen.
"""

tf = TfidfVectorizer(ngram_range=(1, 2))

tf.fit(tourismid_df['Category'])

tf.get_feature_names_out()

tfidf_matrix = tf.fit_transform(tourismid_df['Category'])
tfidf_matrix.shape

tfidf_matrix.todense()

"""**matriks tf-idf untuk beberapa nama wisata (place_name) dan kategori wisata  (Category).**"""

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=tourismid_df.Place_Name
).sample(6, axis=1).sample(10, axis=0)

"""Dari data diatas bisa kita lihat hubungan nama wisata dengan kategori wisata. 0 artinya tidak memiliki hubungan sedangkan angka yang mendekati 1 maka dapat dipastikan kedua fitur memiliki relasi.

## 6.2 Cosine Similarity

Cosine similarity adalah metrik yang digunakan untuk mengukur kesamaan antara dua vektor. Dalam konteks pemrosesan bahasa alami, vektor ini seringkali merepresentasikan dokumen atau teks.
"""

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

cosine_sim_df = pd.DataFrame(cosine_sim, index=tourismid_df['Place_Name'], columns=tourismid_df['Place_Name'])
print('Shape:', cosine_sim_df.shape)


cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""Dengan consine similarity kita berhasil mengidentifikasi kesamaan antara satu  lokasi wisata dengan lokasi wisata lainnya.

## 6.3 Mendapatkan Rekomendasi
"""

def Rekomendasi_wisata_1(Place_Name, similarity_data=cosine_sim_df, items=tourismid_df[['Place_Name', 'Category', 'City','Price']], k=5):

    index = similarity_data.loc[:,Place_Name].to_numpy().argpartition(
        range(-1, -k, -1))


    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(Place_Name, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

tourismid_df[tourismid_df.Place_Name.eq('Pantai Patihan')]

"""Berdasarkan data diatas maka kita akan mencoba merekomendasikan wisata bahari kepada wisatawan. Selanjutnya kita akan mengecek berapa jumlah destinasi wisata pada kategory bahari."""

print("Jumlah kategori bahari:", tourismid_df[tourismid_df['Category'] == 'Bahari']['Category'].count())

"""Berdasarkan data diatas kita akan set Top-n atau K menjadi 47 ,sehingga kita bisa melihat seberapa banyak model dapat memberikan rekomendasi yang sesuai."""

Rekomendasi_wisata_1('Pantai Patihan',k=47).drop_duplicates()

"""Berdasarkan data diatas, model berhasil memberikan 46 rekomendasi yang sesuai, namun terdapat 1 rekomendasi yang tidak sesuai yaitu Wisata Alam Kalibiru yang merupakan kategori Cagar alam.

## 6.4 Metrik Evaluasi
"""

benar = 46
rekomendasi = 47

precision = benar / rekomendasi

print("Precision:", precision)

"""# 7. Model Development dengan Collaborative Filtering

Sistem rekomendasi penyaringan kolaboratif menggunakan informasi tentang preferensi pengguna di masa lalu, seperti rating yang diberikan pada produk atau konten tertentu, untuk memprediksi item mana yang paling mungkin disukai oleh pengguna di masa depan.

## 7.1 Data Preparation

Melakukan penyandian (*encoding*) fitur `User_Id` ke dalam indeks integer.
"""

user_ids = data_wisata.User_Id.unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

print(user_ids)
print(user_to_user_encoded)
print(user_encoded_to_user)

"""Melakukan penyandian *(encoding)* fitur `Place_Id` ke dalam indeks integer."""

Place_Ids = data_wisata.Place_Id.unique().tolist()
Place_to_Place_encoded = {x: i for i, x in enumerate(Place_Ids)}
Place_encoded_to_Place = {i: x for i, x in enumerate(Place_Ids)}


print(Place_Ids)
print(Place_to_Place_encoded)
print(Place_encoded_to_Place)

"""Memetakan `User_Id` dan `Place_Id` ke dalam masing-masing *dataframe* yang berkaitan."""

data_wisata['User_en'] = data_wisata.User_Id.map(user_to_user_encoded)
data_wisata['Place_en'] = data_wisata.Place_Id.map(Place_to_Place_encoded)

"""Melakukan pengecekan jumlah user, jumlah wisata, dan rating minimal serta rating maksimal."""

Num_Users = len(user_encoded_to_user)
Num_Place = len(Place_encoded_to_Place)
print(Num_Users)
print(Num_Place)

Rating_min = min(data_wisata.Place_Ratings)
Rating_max = max(data_wisata.Place_Ratings)
print(f'Number of User: {Num_Users}, Number of Place: {Num_Place}, Min Rating: {Rating_min}, Max Rating: {Rating_max}')

"""## 7.2 Training Data and Validation Data Split

Setelah melakukan pemetaan atribut 'User_en' dan 'Place_en' pada dataframe 'data_wisata', data tersebut akan diacak secara random. Tujuannya adalah untuk memastikan bahwa data yang digunakan dalam analisis selanjutnya tidak memiliki bias akibat urutan data aslinya.
"""

data_wisata = data_wisata.sample(frac=1, random_state=412)
data_wisata

"""Untuk membangun dan mengevaluasi model yang baik, dataset akan dibagi menjadi data latih (80%) dan data uji (20%). Data latih digunakan untuk mengajarkan model mengenali pola dalam data, sementara data uji digunakan untuk mengukur seberapa baik model tersebut dapat memprediksi data yang belum pernah dilihat sebelumnya."""

x = data_wisata[['User_en', 'Place_en']].values
y = data_wisata['Place_Ratings'].apply(lambda x: (x-Rating_min) / (Rating_max-Rating_min)).values

train_indices = int(0.8 * data_wisata.shape[0])

xTrain, xVal, yTrain, yVal = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""## 7.3 Model Development and Training

Untuk membangun model rekomendasi, kita akan memanfaatkan kemampuan deep learning melalui kelas `RecommenderNet` yang disediakan oleh Keras.
"""

class RecommenderNet(tf.keras.Model):
    def __init__(self, Num_Users, Num_Place, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.Num_Users = Num_Users
        self.Num_Place = Num_Place
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            Num_Users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.user_bias      = layers.Embedding(Num_Users, 1)

        self.Place_embedding = layers.Embedding(
            Num_Place,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.Place_bias = layers.Embedding(Num_Place, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0])
        user_bias   = self.user_bias(inputs[:, 0])
        Place_vector = self.Place_embedding(inputs[:, 1])
        Place_bias   = self.Place_bias(inputs[:, 1])

        dot_user_Place = tf.tensordot(user_vector, Place_vector, 2)

        x = dot_user_Place + user_bias + Place_bias

        return tf.nn.sigmoid(x)

"""Untuk melatih model, kita akan menggunakan optimizer Adam yang efisien. Fungsi kehilangan binary crossentropy akan membantu model belajar meminimalkan kesalahan prediksi, sementara metrik RMSE akan digunakan untuk mengukur seberapa akurat model dalam membuat prediksi numerik."""

model = RecommenderNet(Num_Users, Num_Place, 30)

model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss      = BinaryCrossentropy(),
    metrics   = [RootMeanSquaredError()]
)

"""Langkah berikutnya, memulai proses training."""

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    x               = xTrain,
    y               = yTrain,
    batch_size      = 64,
    epochs          = 100,
    validation_data = (xVal, yVal),
    callbacks=[early_stopping],
)

"""**Visualisasi Metrik**"""

rmse     = history.history['root_mean_squared_error']
val_rmse = history.history['val_root_mean_squared_error']

loss     = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (12, 4))
plt.subplot(1, 2, 1)
plt.plot(rmse,     label='RMSE')
plt.plot(val_rmse, label='Validation RMSE')
plt.title('Training and Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Root Mean Squared Error')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(loss,     label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

"""Dari hasil plot, terlihat bahwa model mengalami overfitting. Hal ini ditandai dengan penurunan nilai loss pada data training, sedangkan nilai loss pada data validasi cenderung meningkat. Grafik RMSE menunjukkan hal yang sama, dimana RMSE pada data training terus menurun, sedangkan RMSE pada data validasi cenderung stagnan atau bahkan meningkat. Dengan demikian model perlu dikembangkan lagi kedepannya. Namun dengan nilai RMSE 0,3 seharusnya sudah bisa memprediksi dengan baik. Selanjutnya kita akan menguji apakah model bisa memprediksi dengan baik atau tidak.

## 7.4 Mendapatkan Rekomendasi Wisata

Untuk mendapatkan rekomendasi wisata yang akan dihasilkan oleh sistem, diperlukan sebuah data atau sampel dari pengguna secara acak dan mendefinisikan variabel Place_Name yang belum pernah dibaca oleh pengguna (notVisitedPlace) yang merupakan daftar wisata yang nantinya akan direkomendasikan. Daftar tersebut dapat didapatkan dengan menggunakan operator logika bitwise (~) pada variabel Place_Name yang telah dikunjungi oleh pengguna (VisitedPlace).
"""

userId      = data_wisata.User_Id.sample(1).iloc[0]
VisitedPlace = data_wisata[data_wisata.User_Id == userId]

notVisitedPlace = data_wisata[~data_wisata['Place_Id'].isin(VisitedPlace.Place_Id.values)]['Place_Id']
notVisitedPlace = list(
    set(notVisitedPlace).intersection(set(Place_to_Place_encoded.keys()))
)

notVisitedPlace = [[Place_to_Place_encoded.get(x)] for x in notVisitedPlace]
userEncoder    = user_to_user_encoded.get(userId)
userPlaceArray = np.hstack(
    ([[userEncoder]] * len(notVisitedPlace), notVisitedPlace)
)

"""Untuk mendapatkan rekomendasi daftar wisata, kita akan memanfaatkan fungsi `predict()` dari library Keras."""

ratings = model.predict(userPlaceArray).flatten()

topRatingsIndices   = ratings.argsort()[-10:][::-1]
recommendedPlaceIds = [
    Place_encoded_to_Place.get(notVisitedPlace[x][0]) for x in topRatingsIndices
]

print('Showing recommendations for users: {}'.format(userId))
print('=====' * 8)
print('Place with high ratings from user')
print('-----' * 8)

topPlaceUser = (
    VisitedPlace.sort_values(
        by = 'Place_Ratings',
        ascending=False
    )
    .head(5)
    .Place_Id.values
)

PlaceDfRows = tourismid_df[tourismid_df['Place_Id'].isin(topPlaceUser)]
for row in PlaceDfRows.itertuples():
    print(row.Place_Name, ':', row.Category)

print('=====' * 8)
print('Top 10 Place Recommendation')
print('-----' * 8)

recommended_place = tourismid_df[tourismid_df['Place_Id'].isin(recommendedPlaceIds)]
for row in recommended_place.itertuples():
    print(row.Place_Name, ':', row.Category)
# Laporan Proyek Machine Learning - Doli Sawaluddin

## Project Overview
![Kemenpar Perbanyak Informasi Pariwisata via Media Digital - Lifestyle  Liputan6.com](https://cdn0-production-images-kly.akamaized.net/03d-Lc2TmfRGWdlFqNHVZ7eLvjQ=/800x450/smart/filters:quality(75):strip_icc():format(webp)/kly-media-production/medias/1677197/original/058981000_1502541833-Kemenpar_7_Mei__8__OK.jpg)

Sumber: Liputan6.com

Dilansir dari situs resmi Kementerian Pariwisata dan Ekonomi Kreatif/Badan Pariwisata dan Ekonomi Kreatif (Kemenparekraf/Baparekraf), destinasi pariwisata merupakan inti utama dari pembangunan pariwisata. Dalam pengembangannya, daya tarik wisata sebaiknya dibangun secara sinergis dengan memerhatikan fasilitas wisata, fasilitas umum, dan aksesibilitas/sarana prasarana.[\[1\]](https://kemenparekraf.go.id/ragam-pariwisata/Panduan-Potensi-Pembangunan-Sektor-Pariwisata-dan-Ekonomi-Kreatif)

Berdasarkan pernyataan tersebut bisa kita simpulkan bahwasanya fasilitas dan sarana prasarana merupakan aspek penting dalam industri pariwisata. Namun jika kita teliti lebih dalam tentang potensi pembangunan pemasaran pariwisata, pembangunan destinasi wisata yang sukses tidak hanya bergantung pada daya tarik wisata, fasilitas, dan aksesibilitas yang memadai, tetapi juga pada kemampuan untuk menarik minat wisatawan. Sinergi antara pengembangan infrastruktur, promosi yang efektif, dan pemahaman terhadap preferensi pasar menjadi kunci dalam menciptakan destinasi wisata yang berkelanjutan dan mampu bersaing di tingkat global.

Tak jarang beberapa destinasi wisata yang di masa pembukaan sangat ramai, namun beberapa bulan/tahun kemudian bisa mati dan terbengkalai. Ada banyak penyebab redupnya bisnis pariwisata dan berkurangnya kunjungan wisatawan, salah satu penyebab wisatawan tidak berkunjung  adalah karena bosan atau tidak tertarik dengan destinasi tersebut. Tanpa adanya minat wisatawan, perbaikan dan peningkatan fasilitas sarana dan prasarana wisata hanya akan membuang-buang pengeluaran yang tidak berarti. Maka dari itu merekomendasikan destinasi wisata yang sesuai untuk orang yang tepat adalah langkah yang paling tepat untuk meningkatkan potensi kunjungan. 

Sebenarnya telah banyak penelitian yang dilakukan untuk merekomendasikan destinasi wisata. Faurina et al. misalnya, dimana pada penelitannya yang berjudul 'Implementasi Metode Content-Based Filtering dan Collaborative Filtering pada Sistem Rekomendasi Wisata di Bali', dengan memanfaatkan metode Content-Based Filtering berhasil memberikan rekomendasi berdasarkan preferensi pengguna terhadap kategori destinasi wisata dan  Collaborative Filtering (CF) sendiri menunjukkan hasil performa loss sebesar 0.0589 dan RMSE sebesar 0.2427.[\[2\]](https://publikasi.dinus.ac.id/index.php/technoc/article/view/8556) Hasil penelitian tersebut bisa dikatakan cukup baik dalam membangun model yang dapat memberikan rekomendasi yang dipersonalisasi sesuai minat pengguna.

Berdasarkan Pemaparan diatas, Sistem rekomendasi akan sangat membantu pengusaha/investor dalam memahami minat dan preferensi wisatawan, Dengan memberikan rekomendasi yang sesuai kita dapat mendatangkan wisatawan yang benar-benar tertarik dengan destinasi wisata tertentu. Sehingga diharapkan dapat meningkatkan kunjungan baru , menghindari promosi kepada orang yang kurang tepat dan tentunya menghemat biaya promosi dan pengembangan destinasi wisata.

Pada proyek sistem rekomendasi ini, penulis mencoba memberikan rekomendasi berdasarkan minat pengguna dengan memberikan rekomendasi wisata yang mirip dengan wisata yang pernah dikunjungi, Tidak hanya itu kita juga akan mencoba memberikan rekomendasi yang mungkin disukai oleh wisatawan berdasarkan rating pengguna pada destinasi wisata.



## Business Understanding
### Problem Statements

Berdasarkan latar belakang diatas maka didapatkan rumusan permasalahan sebagai berikut:

 1. Bagaimana cara membangun model machine learning yang dapat memberikan rekomendasi wisata yang mirip dengan wisata yang pernah dikunjungi pengguna sebelumnya?
 2. Bagaimana cara membangun model machine learning yang dapat memberikan rekomendasi yang mungkin disukai wisatawan berdasarkan rating yang diberikan?


### Goals
Berdasarkan pernyataan masalah diatas maka dapat kita tentukan tujuan sebagai berikut:

 1. Membangun model machine learning untuk sistem rekomendasi yang dapat memberikan rekomendasi wisata yang mirip dengan wisata yang pernah dikunjungi menggunakan metode Content Based Filtering.
 2. Membangun model machine learning untuk sistem rekomendasi yang dapat memberikan rekomendasi wisata yang mungkin disukai oleh wisatawan berdasarkan rating yang diberikan menggunakan metode Collaborative Filtering.


### Solution statements
Dari pemaparan sebelumnya, maka terdapat beberapa solusi yang bisa kita gunakan untuk mencapai tujuan dari proyek ini, yaitu:

#### **1. Membangun model machine learning menggunakan metode Content Based Filtering**

Content-Based Filtering adalah metode yang dapat menghasilkan rekomendasi yang bersifat independen kepada pengguna. Content-based Filtering merekomendasikan item yang mirip dengan item lainnya yang sesuai dengan peminatan pengguna.[\[3\]](https://lintar.untar.ac.id/repository/penelitian/buktipenelitian_10390001_7A281222103549.pdf) . Dalam proyek ini kita akan mencoba menggunakan metode ini untuk merekomendasikan wisata yang mirip dengan wisata yang pernah dikunjungi sebelumnya oleh wisatawan.

Pada pendekatan menggunakan _content-based filtering_ kita akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity. Algoritma TF-IDF Vectorizer digunakan untuk mewakili item-item sebagai vektor numerik, sedangkan Cosine Similarity digunakan untuk menghitung kemiripan antara vektor-vektor tersebut.

**- TF-IDF Vectorizer**
Metode Term Frequency-Inverse Document Frequency (TF-IDF) adalah salah satu teknik yang digunakan dalam pengolahan teks dan pemodelan bahasa alami. Tujuan utama dari metode TF-IDF adalah untuk mengevaluasi seberapa penting suatu kata (term) dalam sebuah dokumen dalam konteks koleksi dokumen yang lebih besar. Term Frequency (TF) akan mengukur seberapa sering suatu kata muncul dalam sebuah dokumen. Sedangkan Inverse Document Frequency (IDF) mengukur seberapa penting suatu kata dalam konteks koleksi dokumen yang lebih besar.[\[4\]](https://journal.unj.ac.id/unj/index.php/SINTESIA/article/view/39364) TF-IDF dapat dihitung menggunakan rumus sebagai berikut:

 $$idf_i=log \left( \frac{n}{df_i} \right)$$
 
Keterangan:
	- .
	- $idf_i$ (*Inversed Document Frequency*) merupakan skor IDF untuk *term* $i$; 
	- $df_i$ merupakan banyaknya dokumen yang mengandung *term* $i$; 
	- $n$ merupakan total dokumen. 
	
Semakin tinggi nilai $df$ suatu *term*, maka semakin rendah $idf$ untuk *term* tersebut. Ketika jumlah $df$ sama dengan $n$ yang berarti istilah/*term* tersebut muncul di semua dokumen, $idf$ akan menjadi 0, karena $log(1)=0$. 

Sedangkan nilai TF-IDF merupakan perkalian dari matriks frekuensi *term* dengan IDF-nya.

$$w_{i,j}=tf_{i,j} \times idf_i$$

Di mana $w_{i,j}$ merupakan skor TF-IDF untuk *term* $i$ pada dokumen $j$;  sedangkan $tf_{i,j}$ merupakan frekuensi *term* untuk *term* $i$ pada dokumen $j$, dan $idf_i$ adalah skor $idf$ untuk *term* $i$.
Dengan memanfaatkan TF-IDF kita akan mencoba menemukan kesesuaian antara lokasi wisata dengan kategori wisata.

 **- Cosine Similarity**
Cosine Similarity merupakan metrik yang mengukur kosinus sudut antara dua vektor dimana semakin kecil sudut yang dihasilkan, semakin mirip kedua vektor tersebut. Misalkan sudut antara dua vektor adalah 90 derajat, maka Cosine Similarity akan bernilai 0, ini berarti bahwa kedua vektor saling tegak lurus yang berarti mereka tidak memiliki kemiripan. Sebaliknya, jika Cosine Similarity mendekati atau bahkan bernilai 1, maka sudut antara dua vektor menjadi lebih kecil sehingga mereka akan lebih mirip satu sama lain.[\[5\]](https://medium.com/@ansctrwhyn/penerapan-cosine-similarity-pada-k-nearest-neighbor-5a4f96a6fe90)

Cosine Similarity dilakukan dengan menghitung perkalian skalar antara dua vektor dibagi dengan perkalian panjang vektor keduanya menggunakan rumus sebagai berikut:

$$\cos\theta = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$$
$$\|\vec{a}\| = \sqrt{a_1^2 + a_2^2 + a_3^2 + \cdots + a_n^2}$$
$$\|\vec{b}\| = \sqrt{b_1^2 + b_2^2 + b_3^2 + \cdots + b_n^2}$$

Keterangan:
dimana masing-masing _a_ dan _b_ merupakan vektor dalam ruang multidimensi. Nilai _cos_(𝛳) yang dihasilkan antara -1 sampai 1, sehingga:  
		- Nilai -1 menunjukkan bahwa kedua vektor sangat berlawanan atau tidak memiliki kemiripan.  
		- Nilai 0 menunjukkan vektor ortogonal atau saling tegak lurus.  
		- Nila 1 menunjukkan bahwa kedua vektor memiliki kemiripan yang tinggi.[\[5\]](https://medium.com/@ansctrwhyn/penerapan-cosine-similarity-pada-k-nearest-neighbor-5a4f96a6fe90)

Dengan Consine Similarity kita akan mencoba menemukan kemiripan antara satu destinasi wisata dengan destinasi wisata lainnya.



#### **2. Membangun model machine learning menggunakan metode Collaborative Filtering**
Collaborative Filtering adalah suatu metode dalam pengambilan informasi yang merekomendasikan item kepada pengguna berdasarkan bagaimana pengguna lain dengan preferensi dan perilaku serupa telah berinteraksi dengan item tersebut. Dengan kata lain, algoritma collaborative filtering mengelompokkan pengguna berdasarkan perilaku mereka dan menggunakan karakteristik umum kelompok untuk merekomendasikan item kepada pengguna target.
Sistem rekomendasi kolaboratif beroperasi berdasarkan prinsip bahwa pengguna yang serupa (berdasarkan perilaku) memiliki minat dan selera yang serupa.[\[6\]](https://www.ibm.com/topics/collaborative-filtering)
 
 Colaborative filtering terbagi ke dalam 2 jenis yaitu:
 
 **a. Memory - Based**, Merupakan perluasan dari klasifikasi k-nearest neighbor. Sistem ini berusaha memprediksi perilaku pengguna target terhadap suatu item berdasarkan pengguna serupa atau kumpulan item yang serupa. Sistem berbasis memori dapat dibagi menjadi dua subtipe:
	 -	`User-Based Collaborative Filtering`, Sistem ini merekomendasikan item kepada pengguna target berdasarkan preferensi pengguna lain yang serupa.
	 -	`Item-Based Collaborative Filtering`, Sistem ini merekomendasikan item kepada pengguna target berdasarkan kesamaan antara item-item yang telah disukai oleh pengguna tersebut.

 **b.	Model - Based** , Sistem ini menggunakan model statistik atau machine learning untuk memprediksi preferensi pengguna. Model ini dibangun berdasarkan data interaksi pengguna dengan item-item. Setelah model dilatih, sistem dapat memberikan rekomendasi yang lebih akurat dan efisien dibandingkan dengan metode berbasis memori. Sistem berbasis model dapat dibagi menjadi beberapa subtipe:
	- `Matrix Factorization`: Mengurai matriks rating menjadi dua matriks laten yang merepresentasikan preferensi pengguna dan karakteristik item.
	- `Neural Networks`: Model neural network seperti autoencoder atau deep neural networks dapat digunakan untuk mempelajari representasi laten yang kompleks dari pengguna dan item.
	- `Bayesian Networks`: Memungkinkan pemodelan probabilistik dari hubungan antara pengguna, item, dan faktor-faktor lain yang relevan.[\[6\]](https://www.ibm.com/topics/collaborative-filtering#f01)

Pada proyek ini kita akan menggunakan Memory - Based untuk merekomendasikan destinasi wisata berdasarkan rating yang diberikan pengguna dengan mencari destinasi yang paling mendekati preferensi wisatawan.

## Data Understanding

Pada proyek ini kita akan menggunakan dataset public yang berasal dari *kaggle* dengan detail sebagai berikut :

|  |  Keterangan|
|--|--|
| Sumber | Kaggle Dataset : [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)  |
| Dataset |`package_tourism.csv`, `tourism_rating.csv`, `tourism_with_id.csv`, `user.csv` |

Berdasarkan data diatas untuk membuat sistem rekomendasi  , kita hanya akan memanfaatkan dataset berikut :

| Dataset | Dataframe  | Keterangan |
|--|--|--|
| user.csv |`user_df`   |Berisi informasi tentang user	 |
| tourism_rating.csv | `ratings_df` |	Berisi Informasi rating yang diberikan user |
| tourism_with_id.csv | `tourismid_df` |	Berisi informasi lokasi wisata |

Alasan kenapa kita tidak menggunakan  `package_tourism.csv` adalah karena kita belum mengetahui package mana yang pernah diambil oleh user, sehingga kita hanya bisa mencocokkan kesesuaian user dan package hanya berdasarkan kota yang dia kunjungi. Selanjutnya kita akan melihat 3 dataset yang akan kita gunakan pada proyek ini, masing-masing data set akan diubah kedalam data frame sebagai berikut:

### **1.  user_df** 

![image](https://github.com/user-attachments/assets/a68ea7e9-fad8-4a43-b632-5753c77a570f)

Dataframe user_df berisi variabel-variabel pada *user.csv* dengan detail  sebagai berikut:

 - `User_id` : Merupakan ID pengenal 
 - `Location` : Merupakan Alamat dari user
 - `Age` : Merupakan  usia dari user
 
 Terdapat 3 kolom dengan total baris data sebanyak 300 data, Dataset ini berisi informasi penting seperti alamat dan umur dari user wisatawan. Data ini akan kita kugunakan untuk menganalisa.

### **2.  ratings_df** 

![image](https://github.com/user-attachments/assets/b04ed06c-41fa-4fdb-bd15-be51eefffffa)

Dataframe ratings_df berisi variabel-variabel pada *tourism_rating.csv* dengan detail sebagai berikut:

 - `User_Id` : Merupakan ID pengenal user
 - `Place_Id` : Merupakan kode ID untuk suatu lokasi wisata
 - `Place_Ratings`: Merupakan rating yang diberikan oleh pengguna terhadap suatu lokasi
 
Dataset ini memiliki 3 kolom dengan total baris data sebanyak 10 ribu, dimana data tersebut merupakan informasi ratings yang diberikan oleh setiap wisatawan.

### **3. tourismid_df** 

![image](https://github.com/user-attachments/assets/83e1bfed-5916-4f54-9032-7b5ea75d98fe)

Dataframe  tourismid_df berisi variabel-variabel pada *tourism_rating.csv* dengan detail sebagai berikut:

 - `Place_Id` : Merupakan kode ID untuk suatu lokasi wisata
 - `Place_Name`: Merupakan nama lokasi destinasi wisata
 - `Description`: Deskripsi singkat tengtang destinasi wisata
 - `Category`: Merupakan kategory wisata
 - `City`: Kota dimana wisata tersebut berada
 - `Price`: Merupakan biaya masuk ke lokasi wisata
 - `Rating`: Berisi rating wisata
 - `Time_Minutes`: Berisi nilai waktu
 - `Coordinate` : Berisi lokasi wisata
 - `Lat`: Lokasi wisata dalam latitude
 - `Long`: Lokasi wisata dalam Longitude
 - `Unnamed: 11`: Merupakan kolom yang tidak memiliki nilai, kolom ini akan dihapus nantinya
 - `Unnamed: 12`: Merupakan kolom yang berisi nilai duplikasi, kolom ini akan dihapus nantinya
 
Dataset ini berisikan 13 kolom dengan total baris data sebanyak 437 data, yang artinya terdapat 437 destinasi wisata pada dataset ini.

### **4. Jumlah Data Penting dari Masing-masing Atribut pada Dataset**
Dengan menggunakan fungsi `.unique().` Kita akan melihat jumlah data atribut penting yang ada pada masing-masing dataframe.

![image](https://github.com/user-attachments/assets/3ce6a516-301f-49ac-afb6-9d409ea198aa)

Dari data diatas kita mendapatkan 300 pengguna, 437 destinasi wisata dalam 5 kota dengan jumlah rating mencapai 10.000.


### **5.  Dataset gabungan ratings_df dan tourismid_df** 
Walaupun dengan data `ratings_df` dan `tourismid_id` saja sudah cukup untuk membuat sistem rekomendasi namun untuk memudahkan dalam membaca data, kita akan menggabungkan dataframe `ratings_df` dan `tourismid_df` kedalam variabel `data_wisata`. Berikut adalah tampilan akhir data bersih yang telah disatukan:

![image](https://github.com/user-attachments/assets/5dc355b1-d70b-43dc-9e04-7d7440ed05a1)

 Nantinya setelah dilakukan pembersihan data, kita akan mendapatkan data akhir sebanyak 9921 data baru seperti yang ditampilkan pada gambar diatas.


### **6. Univariate Exploratory Data Analysis (EDA)**
Tahapan Selanjutnya setelah dataset sudah bersih adalah melakukan Exploratory Data Analysis.

####  6.1. Dataset user

**Deskripsi Statistik**

![image](https://github.com/user-attachments/assets/fa61d382-f73d-4965-a29c-72732f73c79d)

Berdasarkan data diatas terdapat 300 pengguna dengan detail sebagai berikut:
- **count:** Jumlah total data yang dianalisis. Dalam hal ini, ada 300 data untuk kedua variabel, yang berarti ada 300 pengguna dalam dataset ini.
- **mean:** Rata-rata atau nilai tengah dari semua data. Rata-rata usia pengguna adalah 28 tahun.
- **std:** Standar deviasi, yang merupakan ukuran sebaran data dari rata-rata. Semakin kecil nilai standar deviasi, semakin dekat data ke nilai rata-rata. Dalam hal ini, standar deviasi usia adalah 6, yang berarti usia pengguna cenderung tersebar dalam rentang 6 tahun di sekitar rata-rata 28 tahun.
- **min:** Nilai minimum. Usia pengguna termuda adalah 18 tahun.
- **25%:** Kuartil pertama. Artinya, 25% dari pengguna berusia 24 tahun atau lebih muda.
- **50%:** Median atau kuartil kedua. Ini adalah nilai tengah dari data yang telah diurutkan. Jadi, 50% pengguna berusia 29 tahun atau lebih muda.
- **75%:** Kuartil ketiga. Artinya, 75% dari pengguna berusia 34 tahun atau lebih muda.
- **max:** Nilai maksimum. Usia pengguna tertua adalah 40 tahun.

**Menampilkan wisatawan berdasarkan daerah asal**

![image](https://github.com/user-attachments/assets/7ea11470-23e4-446a-be42-a7e0e2e4b3aa)

Dari data diatas dapat diambil kesimpulan, wisatawan yang memiliki potensi untuk liburan adalah dari daerah Bekasi, Jawa Barat. dan jumlah asal wisatawan dari rentang 10 sampai 22 juga memiliki potensi berkunjung ke destinasi wisata untuk liburan jika kita bisa merekomendasikan destinasi wisata yang sesuai kepada mereka.

#### 6.2. Dataset tourism_with_id
**Menampilkan jumlah kategori wisata**

![image](https://github.com/user-attachments/assets/f20263fd-560d-45fe-b495-0e2fe5bcb6ea)

Berdasarkan jumlah kategori wisata diatas, diketahui wisata paling diminati adalah wisata taman hiburan, budaya dan cagar alam.

**Deskripsi Statistik**

![image](https://github.com/user-attachments/assets/a2d3959d-5869-48af-ae89-7edc35260ed5)

Bisa kita lihat biaya atau tarif wisata sangat bervariasi mulai dari 0 rupiah (gratis) sampai 900ribu dalam IDR.

#### 6.3. Dataset tourism_rating

**Deskripsi Statistik**

![image](https://github.com/user-attachments/assets/a617327c-3b92-422d-a2c9-eb60016abc31)

Ratings yang diberikan pengguna mulai dari 1 sampai yang tertinggi di angka 5. serta jumlah pengguna yang memberikan rating berjumlah 300 orang dengan total rating mencapai 10 ribu.



## Data Preparation
Data preparation adalah proses mengubah, membersihkan, dan mengorganisasi data agar sesuai dengan kebutuhan model. Melalui data preparation, data yang awalnya tidak terstruktur akan diubah menjadi format yang terstruktur dan siap untuk dianalisis dan digunakan pada tahap pembuatan model machine learning.

### 1. Dataset preparation pada dataframe tourismid_df
Pada tahapan ini bertujuan untuk membersihkan dataframe dari data-data yang tidak dibutuhkan untuk proses pemodelan. Beberapa fitur yang tidak butuhkan dalam dataframe ini bisa kita lihat pada kotak merah:

![image](https://github.com/user-attachments/assets/14b27080-49e7-4b28-81c4-2f19c55f603e)

**Mengecek Nilai yang hilang dan nilai kolom yang duplikat**
Jika kita perhatikan gambar diatas, terdapat missing value pada kolom unnamed:11 dan nilai yang sama antara Place_Id dengan unnamed:12. 

Jumlah missing value pada kolom unnamed:11 ada sebanyak 437 data. Kita dapat mengghapus kedua kolom ini (unnamed 11 dan 12) agar data menjadi bersih. Kita juga akan menghapus time_minutes, cordinate, lat, dan long karena data lokasi dan waktu tidak diperlukan untuk tahapan selanjutnya. Untuk rating kita akan menggunakan rating dari dataframe  `ratings_df`  agar lebih sesuai dengan preferensi pengguna, sehingga kita akan menghapus data rating pada  `tourismid_df`  karena untuk sistem rekomendasi ini  data tersebut tidak akan dipakai. Berikut adalah tampilan data setelah melalui proses penghapusan kolom:

![image](https://github.com/user-attachments/assets/cb58363a-931c-4f0b-a2dc-abcbffd539bf)

### 2. Menggabungkan data rating dan data lokasi wisata
Walaupun dengan data `ratings_df` dan `tourismid_id` saja sudah cukup untuk membuat sistem rekomendasi namun untuk memudahkan dalam membaca data, kita akan menggabungkan dataframe `ratings_df` dan `tourismid_df` kedalam variabel `data_wisata`. Berikut adalah tampilan data setelah disatukan:

![image](https://github.com/user-attachments/assets/111bccd7-6ff4-4d36-a3e1-60a3d5f6c31b)

Kita mendapatkan 10 ribu data baru dan tidak ada perubahan pada jumlah user maupun place name, selanjutnya kita akan melakukan pengecekan kembali terhadap data yang sudah digabungkan pada data preparation, untuk memastikan apakah ada missing value atau duplikasi pada data.

#### **2.1. Mengecek Missing value pada setiap dataframe**

![image](https://github.com/user-attachments/assets/30b2df35-9f5e-4466-bff6-4acfba22dabc)

Berdasarkan data diatas ,ternyata tidak ditemukan adanya missing value pada tiap dataframe

#### 2.2.  Mengatasi duplikasi data
![image](https://github.com/user-attachments/assets/4e405d52-9841-4eda-b82f-10e6170f7fc1)

Berdasarkan data di atas, dapat dilihat bahwa terdapat data duplikat pada data rating dan data gabungan. dimana masing-masing memiliki 79 data duplikat.

**2.2.1. Menghapus duplikasi dan Mengecek kembali jumlah duplikat setelah penghapusan**

![image](https://github.com/user-attachments/assets/2b0516d7-2208-43a8-bb14-598020f58f52)

Sekarang data sudah bersih dengan jumlah akhir sebanyak 9921 baris dengan 10 kolom:

![image](https://github.com/user-attachments/assets/502d3955-4c11-474e-b701-160628035f26)


## Modeling and Result
Pada tahapan ini kita akan membangun model machine learning yang dapat digunakan sebagai sistem rekomendasi destinasi wisata kepada wisatawan tertentu. Ada dua (2) pendekatan model yang akan kita gunakan yaitu Model Development dengan Content Based Filtering dan Model Development dengan Collaborative Filtering. Berikut adalah pemaparan terkait 2 model tersebut:

### 1. Model Development dengan Content Based Filtering
Content-based filtering Memberikan rekomendasi berdasarkan kemiripan atribut dari item atau barang yang disukai oleh pengguna.  [[7]](https://mti.binus.ac.id/2020/11/17/sistem-rekomendasi-content-based/)

Dengan sistem rekomendasi berbasis konten, pengguna akan mendapatkan saran yang lebih personal dan sesuai dengan minat mereka. 

**Kelebihan :**
 - Rekomendasi sangat personal karena didasarkan pada preferensi individu terhadap atribut spesifik destinasi
 - Dapat merekomendasikan item baru yang memiliki atribut serupa dengan item yang disukai pengguna, meskipun item tersebut belum pernah dinilai oleh pengguna lain.
 
**Kekurangan:**
 - `Cold start problem`: Sulit merekomendasikan item kepada pengguna baru yang belum memiliki riwayat interaksi.
 - Terkadang rekomendasi terlalu spesifik dan tidak mengeksplorasi pilihan yang lebih luas.
 - Kualitas rekomendasi sangat bergantung pada kualitas dan relevansi deskripsi item.
 
Setelah mengetahui apa itu Content Based Filtering serta kelebihan dan kekurangannya, kita akan melanjutkan ketahap pemodelan sebagai berikut:

#### 1.1. TF-IDF Vectorizer
TF-IDF digunakan untuk mengubah teks menjadi vektor numerik yang merepresentasikan pentingnya setiap kata dalam dokumen. 

**matriks tf-idf untuk beberapa nama wisata (place_name) dan kategori wisata (Category).**

![image](https://github.com/user-attachments/assets/c2e69cd3-4292-4dc1-bb03-175248a1ae04)

Dari data diatas bisa kita lihat hubungan nama wisata dengan kategori wisata. 0 artinya tidak memiliki hubungan sedangkan angka yang mendekati 1 maka dapat dipastikan kedua fitur memiliki relasi.

#### 1.2. Cosine Similarity
Cosine similarity adalah metrik yang digunakan untuk mengukur kesamaan antara dua vektor. Dalam konteks pemrosesan bahasa alami, vektor ini seringkali merepresentasikan dokumen atau teks.
Berikut adalah hasil dari Consine Similarity :

![image](https://github.com/user-attachments/assets/020a443e-293c-4807-9693-d5a035b1a7ca)

Dengan consine similarity kita berhasil mengidentifikasi kesamaan antara satu lokasi wisata dengan lokasi wisata lainnya. Bisa kita lihat pada gambar angka 1 menunjukkan kecocokan antara satu wisata dengan wisata lainnya, sedangkan angka 0 menunjukkan tidak adanya kemiripan pada kedua lokasi wisata. Contohnya  Hutan Pinus Asri sangat mirip dengan Hutan Wisata Tinjomoyo Semarang.

#### 1.3. Hasil Top-N Recommendation
Berikut adalah hasil pengujian sistem rekomendasi dengan pendekatan `content-based recommendation`:

![image](https://github.com/user-attachments/assets/ae724778-7ea5-4bf5-a871-531e9a6c8ab2)

Data diatas merupakan data destinasi wisata yang dipilih oleh wisatawan. Berdasarkan data tersebut kita akan merekomendasikan destinasi wisata yang mirip dengan **Pantai Patihan**. Berikut adalah hasil rekomendasinya:

![image](https://github.com/user-attachments/assets/4a8ab971-7a78-47bc-be34-8bd3eb4722c8)

Alasan kenapa menampilkan top-n sampai 47 adalah agar kita bisa melihat kesalahan pada model. Berdasarkan data diatas, model berhasil memberikan 46 rekomendasi yang sesuai, namun terdapat 1 rekomendasi yang tidak sesuai yaitu Wisata Alam Kalibiru yang merupakan kategori Cagar alam. Namun untuk penerapan model pada aplikasi sistem rekomendasi yang sebenarnya kita hanya perlu menyesuaikan jumlah rekomendasi yang ditampilkan menjadi 3 - 5 rekomendasi atau sesuai kebutuhan.

### 2. Model Development dengan Collaborative Filtering
Sistem rekomendasi penyaringan kolaboratif menggunakan informasi tentang preferensi pengguna di masa lalu, seperti rating yang diberikan pada produk atau konten tertentu, untuk memprediksi item mana yang paling mungkin disukai oleh pengguna di masa depan. Berikut adalah kelebihan dan kekurangan model ini:

**Kelebihan :**
 - Dapat menemukan item yang mungkin tidak terpikirkan oleh pengguna, tetapi disukai oleh pengguna lain dengan preferensi serupa.
 - Tidak memerlukan deskripsi rinci tentang item, hanya bergantung pada pola interaksi pengguna seperti ratings.
 - Akurasi rekomendasi akan meningkat seiring dengan bertambahnya data interaksi pengguna.
 
**Kekurangan:**
 - Cold start problem: Sama seperti CBF, sulit merekomendasikan item kepada pengguna baru atau item baru yang belum banyak dinilai.
 - Dapat menjadi kompleks secara komputasi untuk sistem dengan jumlah pengguna dan item yang sangat besar.
 
 Setelah mengetahui apa itu Collaborative Filtering serta kelebihan dan kekurangannya, kita akan melanjutkan ketahap pemodelan sebagai berikut:

#### 2.1. Data Preparation
Pada tahapan ini kita melakukan penyandian (_encoding_) fitur `User_Id`dan `Place_Id` pada data frame data wisata ke dalam indeks integer dengan hasil sebagai berikut:

**encoding User_Id :**

![image](https://github.com/user-attachments/assets/c28e8f3a-092c-4fd0-a81d-2d11f8a7059b)

**encoding Place_Id**
![image](https://github.com/user-attachments/assets/ab52e224-edf6-4df9-8199-4a22942ab639)

Selanjutnya setelah melakukan encoding maka kita akan memetakan `User_Id` sebagai User_en dan `Place_Id` sebagai Place_en ke dalam dataframe data_wisata. 
Setelah melakukan tahapan diatas Diperoleh jumlah _user_ sebesar 300, jumlah wisata sebesar 437, nilai minimal _rating_ yaitu 1, dan nilai maksimum _rating_ yaitu 5.

#### 2.2. Training Data and Validation Data Split
Setelah melakukan pemetaan atribut 'User_en' dan 'Place_en' pada dataframe 'data_wisata', data tersebut akan diacak secara random. Tujuannya adalah untuk memastikan bahwa data yang digunakan dalam analisis selanjutnya tidak memiliki bias akibat urutan data aslinya.

![image](https://github.com/user-attachments/assets/3804801f-a96b-4589-9210-bdfa16a94368)

Selanjutnya Untuk membangun dan mengevaluasi model yang baik, dataset akan dibagi menjadi data latih (80%) dan data uji (20%). Data latih digunakan untuk mengajarkan model mengenali pola dalam data, sementara data uji digunakan untuk mengukur seberapa baik model tersebut dapat memprediksi data yang belum pernah dilihat sebelumnya.


#### 2.3. Model Development and Top-n Result
Untuk membangun model rekomendasi, kita akan memanfaatkan kemampuan deep learning melalui kelas `RecommenderNet` yang disediakan oleh Keras. Sedangkan Untuk melatih model, kita akan menggunakan optimizer Adam yang efisien serta untuk mengukur kinerja model kita akan menggunakan metrik RMSE.

**Mendapatkan  Rekomendasi Wisata**
Berikut adalah hasil rekomendasi yang diberikan oleh sistem:

![image](https://github.com/user-attachments/assets/058ed447-0897-4bb2-aa1d-0430fbb20507)

Berdasarkan hasil di atas, dapat dilihat bahwa sistem akan mengambil pengguna secara acak, yaitu pengguna dengan `User_Id`  **230**. Lalu akan dicari 5 destinasi wisata dengan _rating_ terbaik dari user tersebut. Dimana ternyata user ini memiliki ketertarikan dengan wisata Bahari, Taman Hiburan, Cagar alam, dan budaya.

Selanjutnya sistem akan membandingkan antara wisata dengan rating tertinggi dari user dengan semua destinasi wisata yang ada, kecuali 5 daftar wisata yang sudah dikunjungi sebelumnya, lalu sistem akan mengurutkan wisata yang akan direkomendasikan berdasarkan nilai rekomendasi yang paling tinggi.

Sampai tahap ini model telah berhasil memberikan prediksi yang baik. Dimana kita bisa melihat sistem merekomendasikan wisata dengan kategori budaya, taman hiburan, cagar alam , dan Bahari berdasarkan beberapa wisata dengan rating tinggi yang pernah dikunjungi oleh pengguna. Namun terdapat 1 rekomendasi yang salah yaitu sistem merekomendasikan Tempat Ibadah, kesalahan ini merupakan hal yang wajar dalam prediksi mengingat prediksi dipengaruhi oleh akurasi model.



## Evaluation

#### **1. Content-based Recommendation**
Ketika kita membangun sebuah model rekomendasi berbasis konten, langkah evaluasi sangat penting untuk mengukur seberapa baik model tersebut bekerja. Untuk model ini Kita akan menggunakan metrik Precision, Precision adalah metrik yang mengukur seberapa akurat sebuah sistem rekomendasi dalam memberikan rekomendasi yang relevan. Dalam konteks protek ini, precision menunjukkan seberapa besar persentase tempat wisata yang direkomendasikan yang memang sesuai dengan preferensi pengguna. Precision dapat dihitung menggunakan rumus berikut:

$$Presisi = \frac{\text{Jumlah rekomendasi yang relevan}}{\text{Jumlah total rekomendasi}}$$
   
Berdasarkan hasil rekomendasi model yang telah dibuat maka didapat Precission dengan detail sebagai berikut:

![image](https://github.com/user-attachments/assets/bf2a120d-9591-4ab0-b9b8-43f5602045b0)

Hasil Precision yang kita dapat adalah 0,97%.


#### **2. Collaborative Filtering Recommendation**  
Seperti yang telah dibahas sebelumnya metrik yang digunakan untuk model adalah metrik RMSE. 
RMSE adalah metrik yang digunakan untuk mengukur rata-rata perbedaan antara nilai prediksi dan nilai aktual. Rumus RMSE adalah sebagai berikut: 

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$ 

Di mana: 
* $n$: Jumlah data 
* $y_i$: Nilai aktual ke-i 
* $\hat{y}_i$: Nilai prediksi ke-i 
* Nilai RMSE yang lebih kecil menunjukkan model yang lebih akurat.


Metrik RMSE akan digunakan untuk mengukur seberapa akurat model dalam membuat prediksi numerik. Setelah model dibangun dan dilatih kita dapat melihat visualisasi metriknya sebagai berikut:

![image](https://github.com/user-attachments/assets/363a522a-50d7-400e-808e-e4406f7ef42d)

Dari hasil plot, terlihat bahwa model mengalami sedikit overfitting. Hal ini ditandai dengan penurunan nilai loss pada data training, sedangkan nilai loss pada data validasi cenderung meningkat. Grafik RMSE menunjukkan hal yang sama, dimana RMSE pada data training terus menurun, sedangkan RMSE pada data validasi cenderung stagnan atau bahkan meningkat. Dengan demikian model perlu dikembangkan lagi kedepannya. Namun dengan nilai RMSE 0,33 seharusnya sudah bisa memprediksi data dengan baik.


### Kesimpulan

Berdasarkan hasil model yang telah kita bangun menggunakan Content Based Filtering dan Collaborative Filtering kita mendapatkan metrik precission dan RMSE yang cukup baik. dan kita telah berhasil Membangun model yang dapat memberikan rekomendasi wisata yang mirip dengan wisata yang pernah dikunjungi serta destinasi yang mungkin disukai oleh wisatawan berdasarkan rating yang pernah mereka berikan.

Dengan adanya sistem rekomendasi ini, diharapkan dapat menarik minat wisatawan untuk berkunjung dan liburan kembali. Namun tentunya rekomendasi yang baik bukan hanya sampai disini saja, setelah kita memberikan rekomendasi yang sesuai, kita juga harus memberikan informasi dan fasilitas yang relevan untuk wisatawan, sehingga sektor pariwisata akan dapat bertahan dan terus berkembang kedepannya


## Referensi
[1] Kementerian Pariwisata dan Ekonomi Kreatif/Badan Pariwisata dan Ekonomi Kreatif (Kemenparekraf/Baparekraf). (2021). Panduan Potensi Pembangunan Sektor Pariwisata dan Ekonomi Kreatif. _Kemenparekraf_. Retrived from: https://kemenparekraf.go.id/ragam-pariwisata/Panduan-Potensi-Pembangunan-Sektor-Pariwisata-dan-Ekonomi-Kreatif

[2] Faurina, R., & Sitanggang, E. (2023). Implementasi Metode Content-Based Filtering dan Collaborative Filtering pada Sistem Rekomendasi Wisata di Bali. _Techno.Com_, _22_(4), 870–881. https://doi.org/10.33633/tc.v22i4.8556

[3] Salim, E., Pragantha, J., & Lauro, M. D. (n.d.). Perancangan Sistem Rekomendasi Film menggunakan metode Content- based Filtering. _untar_. Retrived from: https://lintar.untar.ac.id/repository/penelitian/buktipenelitian_10390001_7A281222103549.pdf

[4] Septiani, D., & Isabela, I. (2022). ANALISIS TERM FREQUENCY INVERSE DOCUMENT FREQUENCY (TF-IDF) DALAM TEMU KEMBALI INFORMASI PADA DOKUMEN TEKS. _SINTESIA_, _01_(2), 81-88. Retrived from: https://journal.unj.ac.id/unj/index.php/SINTESIA/article/view/39364

[5] Wahyuni, A. C. (2021). Penerapan Cosine Similarity pada K-Nearest Neighbor. _Medium_. Retrived from: https://medium.com/@ansctrwhyn/penerapan-cosine-similarity-pada-k-nearest-neighbor-5a4f96a6fe90

[6] Murel, J., & Kavlakoglu, E. (2024, 21 Maret). What is collaborative filtering?. _IBM_. Retrived from: https://www.ibm.com/topics/collaborative-filtering

[7] BINUS University. (2020, November 17). Sistem rekomendasi- Content Based. Retrieved from: https://mti.binus.ac.id/2020/11/17/sistem-rekomendasi-content-based/

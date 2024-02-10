# Laporan Proyek Machine Learning - Muhammad Syiarul Amrullah
![Image of Sloth](https://www.travelandleisure.com/thmb/cQ_qSlzajuIUvVE-tckLfWCSBOA=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/sloth-SLOTH1018-2783079be65d4717b73e17f1db1700db.jpg)
## Domain Proyek

Sloth merupakan mamalia arboreal yang dikenal dengan gerakan lambatnya dan gaya hidupnya yang santai. Terdapat dua famili yang terdiri dari enam spesies sloth: Megalonychidae (two-toed sloth) dan Bradypodidae (three-toed sloth). Meskipun sloth memiliki penampilan yang unik dan ciri-ciri khas, namun klasifikasi spesies sloth masih menjadi tantangan karena beberapa faktor, seperti perubahan warna bulu yang dapat membingungkan dalam identifikasi spesies, serta adanya variasi dalam ukuran tubuh dan morfologi antar spesies. terdapat kasus klasifikasi lain seperti [Poverty Classification Using Machine Learning: The Case of Jordan](https://www.semanticscholar.org/paper/Poverty-Classification-Using-Machine-Learning%3A-The-Alsharkawi-Al-Fetyani/7ceaa167d3a41af5477961bb042a246aaceb4b17). penelitian tersebut ditujukan dalam menyelesaikan permasalahan terhadap penilaian dan monitoring kemiskinan di yordania. Maka dari itu klasifikasi spesies sloth dilakukan menggunakan machine learning.

Penelitian klasifikasi sloth memiliki tujuan untuk mengembangkan metode atau algoritma yang dapat mengidentifikasi dan mengklasifikasikan spesies sloth secara akurat berdasarkan karakteristik morfologi atau genetik. Hal ini penting untuk memahami keragaman genetik dan ekologi sloth serta melindungi keberlanjutan populasi mereka. penelitian ini diharapkan dapat memberikan kontribusi dalam pemahaman lebih lanjut tentang sloth dan upaya konservasi mereka.

## Business Understanding
Masalah klasifikasi sloth memiliki dampak penting dalam bidang konservasi dan penelitian ekologi. Dengan memahami lebih baik tentang spesies sloth, kita dapat mengidentifikasi area-area penting untuk pelestarian habitatnya, mengembangkan strategi konservasi yang lebih efektif, dan memastikan kelangsungan hidup populasi sloth di alam liar. Selain itu, pengetahuan yang lebih mendalam tentang sloth juga dapat memberikan wawasan yang berharga dalam bidang biologi evolusi dan kajian ekologi hewan arboreal lainnya.

Dari segi ekonomi, penelitian ini juga dapat memberikan manfaat dalam pengembangan pariwisata berkelanjutan. Sloth sering menjadi daya tarik wisatawan karena keunikan dan keanggunannya, sehingga pemahaman yang lebih baik tentang sloth dapat membantu dalam pengembangan program pariwisata yang bertanggung jawab dan berkelanjutan untuk mendukung ekonomi lokal di daerah-daerah di mana sloth hidup.

Secara keseluruhan, penelitian klasifikasi sloth memiliki implikasi yang luas dalam bidang konservasi, penelitian ekologi, dan pengembangan pariwisata, yang semuanya memiliki dampak positif baik secara ekologis maupun ekonomis.

### Problem Statements
Berdasarkan latar belakang dari penelitian ini, rumusan masalah pada penelitian ini sebagai berikut:
- Bagaimana cara melakukan pra-pemrosesan pada data sloth yang akan digunakan untuk membuat model yang baik? 
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap klasifikasi sloth?
- Klasifikasi apa yang muncul dengan fitur tertentu?

### Goals

Berdasarkan rumusan masalah dari penelitian ini, tujuan pada penelitian ini sebagai berikut:
- Mengetahui tahapan pra pemrosesan data yang tepat dalam klasifikasi sloth
- Mengetahui fitur yang paling berkorelasi dengan klasifikasi sloth
- Membuat model machine learning yang dapat klasifikasi sloth dengan tingkat akurasi yang tinggi

 ### Solution statements
 Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
 - Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Melakukan _drop_ kolom pada kolom Unnamed:0.
    * Melakukan convert datatype object -> category untuk data nominal atau ordinal
    * Melakukan check missing value dan melakukan penanganan missing value
    * Melakukan check outliers value dan melakukan penanganan outliers
    * Melakukan check distribution
    * Mengatasi masalah data tidak seimbang dengan _resample_.
    * Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
    * Melakukan _Standard Scaler_.

* Untuk pembuatan model dipilih penggunaan model dengan algoritma Random Forest dan K-Nearest Neighbor. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. Berikut cara kerja, kelebihan dan kekurangan algoritma Random Forest dan K-Nearest Neighbor:
    * Cara kerja Algoritma Random Forest [[6]](https://repository.usd.ac.id/35513/):
        * Diawali dengan pemilihan k pada sampel dataset yang diambil secara acak dengan pengembalian
        * Gunakan dataset untuk membangun _decision tree_ ke-i
        * Ulangi langkah kedua langkah diatas sebanyak k.
    * Kelebihan dan kekurangan Algoritma Random Forest [[7]](https://eprints.umm.ac.id/39299/):
        * Kelebihannya yaitu dapat mengatasi _noise_ dan _missing value_ serta dapat mengatasi data dalam jumlah yang besar.
        * Kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data. 
    * Cara kerja Algoritma K-Nearest Neighbor [[8]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/):
        * Menentukan jumlah tetangga terdekat K
        * Menghitung jarak dokumen _testing_ ke dokumen _training_
        * Urutkan data berdasarkan data yang mempunyai jarak Euclidean terkecil
        * Tentukan kelompok testing berdasarkan label pada K.
    * Kelebihan dan kekurangan Algoritma K-Nearest Neighbor [[9]](https://simdos.unud.ac.id/uploads/file_penelitian_1_dir/721bdb509a6f0bb9ccca6d7374b86759.pdf):
        * KNN memiliki beberapa kelebihan yaitu bahwa algoritmanya tangguh terhadap _training_ data yang _noisy_ dan efektif apabila data latihnya besar.
        * Kekurangan pada algoritma KKN yaitu perlu menentukan nilai dari parameter K (jumlah dari tetangga terdekat), Pembelajaran berdasarkan jarak tidak jelas mengenai jenis jarak apa yang harus digunakan dan atribut mana yang harus digunakan untuk mendapatkan hasil yang terbaik dan Biaya komputasi cukup tinggi karena diperlukan perhitungan dari jarak tiap sample uji pada keseluruhan sample latih.
  - PCA digunakan untuk mengurangi dimensi sehingga dapat mencegah overfitting pada model
  - Lazy Classification digunakan untuk mendapatkan algoritma dengan tingkat akurasi tinggi
  - Hyperparameter Tuning dilakukan untuk mendapatkan parameter terbaik pada model sehingga mendapatkan hasil terbaik

## Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:

- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

# Laporan Proyek Machine Learning - Muhammad Syiarul Amrullah
Domain yang dipilih untuk proyek machine learning ini adalah Konservasi, dengan judul Predictive Analytics : Klasifikasi Jenis Sloth Menggunakan Machine Learning
![Image of Sloth](https://www.travelandleisure.com/thmb/cQ_qSlzajuIUvVE-tckLfWCSBOA=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/sloth-SLOTH1018-2783079be65d4717b73e17f1db1700db.jpg)
## Domain Proyek

Sloth merupakan mamalia arboreal yang dikenal dengan gerakan lambatnya dan gaya hidupnya yang santai. Terdapat dua famili yang terdiri dari enam spesies sloth: Megalonychidae (two-toed sloth) dan Bradypodidae (three-toed sloth). Meskipun sloth memiliki penampilan yang unik dan ciri-ciri khas, namun klasifikasi spesies sloth masih menjadi tantangan karena beberapa faktor, seperti perubahan warna bulu yang dapat membingungkan dalam identifikasi spesies, serta adanya variasi dalam ukuran tubuh dan morfologi antar spesies. Memahami jenis-jenis sloth dapat membantu dalam upaya konservasi untuk melindungi spesies-spesies yang terancam punah dan memastikan keberlanjutan populasi mereka. terdapat kasus klasifikasi lain seperti [[1]](https://www.semanticscholar.org/paper/Poverty-Classification-Using-Machine-Learning%3A-The-Alsharkawi-Al-Fetyani/7ceaa167d3a41af5477961bb042a246aaceb4b17). penelitian tersebut ditujukan dalam menyelesaikan permasalahan terhadap penilaian dan monitoring kemiskinan di yordania. Maka dari itu klasifikasi sloth ditjujukan untuk menyelesaikan permasalahan dan membantu proses konservasi.

Tujuan untuk mengembangkan metode atau algoritma yang dapat mengidentifikasi dan mengklasifikasikan spesies sloth secara akurat berdasarkan karakteristik morfologi atau genetik. Hal ini penting untuk memahami keragaman genetik dan ekologi sloth serta melindungi keberlanjutan populasi. penelitian ini diharapkan dapat memberikan kontribusi dalam pemahaman lebih lanjut tentang sloth dan upaya konservasi mereka.

## Business Understanding
Masalah klasifikasi sloth memiliki dampak penting dalam bidang konservasi dan penelitian ekologi. Terdapat beberapa masalah yaitu, pemantauan populasi yang kurang efektif, kurangnya pemahaman sehingga kesadaran masyarakat rendah terhadap perlindungan sloth. penelitian ini bertujuan membantu untuk memahami lebih baik tentang spesies sloth. Pemahaman sloth yang baik dapat membantu identifikasi area-area penting untuk pelestarian habitatnya, mengembangkan strategi konservasi yang lebih efektif, dan memastikan kelangsungan hidup populasi sloth di alam liar. Selain itu, pengetahuan yang lebih mendalam tentang sloth juga dapat memberikan wawasan yang berharga dalam bidang biologi evolusi dan kajian ekologi hewan arboreal lainnya.

Dari segi ekonomi, Pemahaman yang lebih baik tentang sloth dapat memberikan dampak positif yang signifikan pada pengembangan program pariwisata yang berkelanjutan. Salah satu aspek utamanya adalah daya tarik wisatawan yang lebih besar. Sloth merupakan hewan yang unik dan menarik bagi banyak orang, sehingga pemahaman yang lebih mendalam tentang sloth dapat menghasilkan program pariwisata yang lebih menarik dan edukatif. Program ini dapat menarik lebih banyak wisatawan untuk mengunjungi daerah tersebut, yang pada gilirannya dapat meningkatkan pendapatan pariwisata dan ekonomi lokal.

Selain itu, pemahaman yang lebih baik tentang sloth juga dapat membantu dalam melindungi habitatnya. Dengan memahami habitat dan perilaku sloth, program pariwisata dapat merancang jalur wisata yang sesuai dan meminimalkan dampak negatif terhadap lingkungan. Program pariwisata yang bertanggung jawab terhadap lingkungan akan membantu dalam menjaga keberlanjutan lingkungan dan habitat sloth.

Pengembangan program pariwisata yang berkelanjutan juga dapat memberikan manfaat dalam hal pendidikan dan kesadaran lingkungan. Melalui program ini, wisatawan dapat diberi informasi tentang pentingnya melindungi sloth dan habitatnya. Hal ini dapat meningkatkan kesadaran masyarakat tentang konservasi lingkungan dan membantu dalam melibatkan masyarakat lokal dalam upaya pelestarian sloth.

Selain itu, program pariwisata yang berkelanjutan juga dapat menjadi sumber pendapatan ekonomi tambahan bagi masyarakat lokal. Dengan meningkatkan kunjungan wisatawan, program pariwisata dapat meningkatkan kesejahteraan masyarakat setempat dan mengurangi tekanan terhadap sumber daya alam. Dengan demikian, pemahaman yang lebih baik tentang sloth dapat berkontribusi pada pengembangan program pariwisata yang bertanggung jawab, berkelanjutan, dan bermanfaat bagi masyarakat lokal dan lingkungan.

Secara keseluruhan, penelitian klasifikasi sloth memiliki implikasi yang luas dalam bidang konservasi, penelitian ekologi, dan pengembangan pariwisata, yang semuanya memiliki dampak positif baik secara ekologis maupun ekonomis.

### Problem Statements
Berdasarkan latar belakang dari penelitian ini, rumusan masalah pada penelitian ini sebagai berikut:
- Bagaimana melakukan pra-pemrosesan pada data sloth yang akan digunakan untuk membuat model yang baik? 
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap klasifikasi sloth?
- Bagaimana modelling dilakukan sehingga dapat meningkatkan pemahaman masyarakat dan membantu monitoring terhadap slot?

### Goals

Berdasarkan rumusan masalah dari penelitian ini, tujuan pada penelitian ini sebagai berikut:
- Mengetahui tahapan pra pemrosesan data yang tepat dalam klasifikasi sloth
- Mengetahui fitur yang paling berkorelasi dengan klasifikasi sloth
- Membuat model machine learning yang dapat klasifikasi sloth dengan tingkat akurasi yang tinggi diatas 90%

 ### Solution statements
 Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
 - Untuk data understanding dan data preparation dapat dilakukan beberapa teknik, diantaranya :
    * Check dan Penanganan Missing Value: Melakukan pemeriksaan terhadap data untuk mengetahui apakah terdapat nilai yang hilang (missing value), dan jika ada, melakukan penanganan seperti mengisi nilai yang hilang atau menghapus baris/data yang memiliki nilai yang hilang tersebut.
    * Memeriksa apakah terdapat nilai ekstrem (outliers) dalam data, dan jika ditemukan, melakukan penanganan seperti menghapus outliers atau mengubah nilainya agar sesuai dengan distribusi data.
    * Memeriksa distribusi data untuk setiap fitur/variabel dalam dataset, dan jika diperlukan, melakukan transformasi data agar distribusinya lebih sesuai dengan asumsi model yang digunakan.
    * Mengubah nilai kategori (categorical value) menjadi nilai numerik agar dapat digunakan dalam analisis atau pemodelan.
    * Menstandarisasi skala variabel numerik agar memiliki skala yang seragam, sehingga mencegah variabel dengan skala besar mendominasi pengaruh pada model.
    * Memisahkan dataset menjadi dua bagian dengan rasio 80% untuk data latih (training data) dan 20% untuk data uji (testing data). Hal ini dilakukan untuk melatih model pada data latih dan menguji performa model pada data uji yang tidak digunakan dalam proses pelatihan.
* Menggunakan TPOT Classifier untuk mendapatkan algoritma machine learning dengan tingkat akurasi yang tinggi secara otomatis. TPOT Classifier akan melakukan pencarian dan evaluasi berbagai kombinasi algoritma serta hyperparameter untuk menemukan model terbaik untuk dataset yang diberikan.
* Melakukan tuning (penyetelan) hyperparameter dari model yang dihasilkan oleh TPOT Classifier untuk mendapatkan parameter terbaik yang menghasilkan performa model yang optimal.

## Data Understanding
Dataset yang digunakan untuk proyek ini diperoleh dari situs kaggle yang dapat diunduh melalui [Kaggle](https://www.kaggle.com/datasets/bertiemackie/sloth-species). 
Dataset ini memiliki 5000 data dan 6 feature + 1 target. Sampel data akan dibagi menjadi dua, *Numerical Features* (size_cm, claw_length_cm, tail_length_cm, dan weight_kg) dan *Categorical Features* (endangered, specie, dan sub_specie).
Adapun penjelasan detail dari sampel data sebagai berikut:
- size_cm: _The size of the sloth (head & body) in cm_.
- claw_length_cm: _The length of the sloths claws in cm_. 
- tail_length_cm: _The length of the sloths tail in cm._
- weight_kg: _The weight of the sloth in kg_.
- endangered: _The endangered category for the sub species._
- specie: _The species of sloth, two or three toed_.
- sub_specie: _The sub specie of sloth._
Pada data understanding, terdapat beberapa tahap dalam memahami data seperti _check & handle Missing Value_, _check & handle outliers_, _check & handle distribution data_.
### check & handle Missing Value
berdasarkan gambar diatas, tidak ditemukan missing value disetiap kolom. selain pengecekan dengan missing value, perlu dilakukan pengecekan terhadap invalid data, penelitian ini menggunakan kondisional jika claw_length_cm/size_cm/tail_length_cm kurang dari 0 untuk menemukan invalid data, karena tidak ada sloth yang memiliki nilai tersebut kurang dari 0. hasilnya sebagai berikut:
![grafik invalid data](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/invalid_data.png?raw=true)

Gambar 1. Invalid data

435 invalid data pada dataset di _drop_, Invalid data perlu dihapus karena dapat mengganggu analisis dan pemodelan data yang akurat. Data yang tidak valid, seperti nilai yang hilang atau format yang salah, dapat menyebabkan kesalahan dalam perhitungan statistik dan menghasilkan hasil yang tidak dapat diandalkan. Dengan menghapus data yang tidak valid membuat data yang bersih dan representatif, sehingga hasilnya lebih akurat dan dapat dipercaya.
### Check & handle outliers
![percentage of outliers](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/percentage%20of%20outliers.png?raw=true)

Gambar 2. Percentage of outliers

berdasarkan gambar diatas, terdapat outliers di beberapa kolom. 25% size_cm, 0.5 weight_kg & 0.25 claw_length_cm. outliers pada size_cm melebihi 10% sehingga data di drop dengan method IQR, dikarenakan outliers yang signifikan seperti ini dapat memiliki dampak yang tidak proporsional terhadap analisis statistik dan pemodelan. Dengan menghapus outliers ini, distribusi data dapat menjadi lebih representatif dan hasil analisis serta model yang dihasilkan akan lebih akurat dan dapat diandalkan. setelah 1072 outliers di drop, berikut hasil setelah penanganan outliers

![after outliers](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/after_outliers.png?raw=true)

Gambar 3. Grafik boxplot after remove outliers
### Distribusi variabel
Distribusi variabel merujuk pada cara nilai-nilai dari suatu variabel terdistribusi atau tersebar dalam sebuah populasi atau sampel. Distribusi ini sering kali digambarkan menggunakan grafik atau fungsi matematika tertentu yang menggambarkan pola sebaran nilai-nilai variabel tersebut. untuk mengetahui persebaran data dari masing-masing feature menggunakan skew(). berikut nilai skew pada masing masing feature:
![distribusi variabel](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/skewness_feature.png?raw=true)

Gambar 4. Skewness each feature
berdasarkan gambar 4, setiap feature terdistribusi normal karena nilai skewness tidak melebihi -4 dan 4. maka dari itu, tidak perlu dilakukan preprocessing terhadap persebaran data

### EDA-Univariate
Berikut _univariate analysis_ terhadap _categorical feature_ dan _numerical feature_
#### Categorical Feature
![univariate categorical feature](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/uni_categorical_features.png?raw=true)

Gambar 5. Univariate categorical feature
#### Numerical Feature
![grafik sex](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/uni_numerical_features.png?raw=true)

Gambar 6. Grafik Sex

Berdasarkan gambar 6, persebaran data setiap feature terdistribusi secara normal dengan keberadaan data di sekitar nilai median.
### EDA-Multivariate
Berikut merupakan *EDA-Multivariate Analysis*:
#### Correlation between feature
#### Categorical Feature
![multivariate endangered](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/endangered_specie.png?raw=true)
![multivariate subspecie](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/sub_specie_specie.png?raw=true)
Gambar 7. Multivariate Feature

Berdasarkan gambar 7, korelasi endangered dengan specie dominan pada satu value yaitu least_concern. korelasi sub specie dengan specie dominan terhadap 2 value yaitu Hoftman two toed dan slothLinnaeus two toed sloth. kedua feature tersebut di drop karena tidak berkorelasi dengan specie.
#### Numerical Feature
![Correlation numerical feature](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/corr_matrix.png?raw=true)

Gambar 8. Correlation numerical feature

Berdasarkan gambar 8, korelasi antara size_cm dan tail length_cm negatif, size_cm dan weight_kg positif, tail_length_cm dan weight_kg memiliki korelasi negatif dan korelasi antara feature yang lain netral, berikut korelasi feature dengan target(specie)
![Correlation numerical feature with target](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/corr_matrix_specie.png?raw=true)

Gambar 9. Correlation numerical feature with target

Hasilnya, size_cm dan weight_kg memiliki korelasi positif dengan specie, dan tail_length_cm memiliki korelasi negatif.
## Data Preparation
Data preparation memiliki tahapan sebagai berikut encoding, Dimension reduction, Scaling, splitting dataset. Berikut penerapan dan hasil dari data preparation:
1. Encoding
Pada tahap ini, encoding dilakukan untuk 1 feature yaitu specie. penerapan encoding menggunakan label encoder karena feature ini merupakan variabel target. Hasilnya, Three toed menjadi 0 dan two toed menjadi 1. Encoding dipilih dalam analisis data untuk mengubah variabel kategori menjadi bentuk numerik yang dapat diproses oleh algoritma machine learning. Penggunaan teknik encoding ini memiliki beberapa tujuan utama, di antaranya adalah untuk menghindari bias dan meningkatkan kinerja model. Encoding membantu menghindari bias dengan mengubah variabel kategori menjadi bentuk yang dapat dimengerti oleh model, sehingga mencegah kesalahan interpretasi atau analisis yang tidak akurat. Selain itu, encoding juga dapat meningkatkan kinerja model dengan memungkinkan model untuk memproses informasi dengan lebih efisien. Dengan mengubah variabel kategori menjadi bentuk numerik, model dapat membuat prediksi yang lebih akurat dan cepat. Teknik encoding juga mendukung penggunaan berbagai algoritma machine learning yang memerlukan input numerik. Dengan menggunakan teknik encoding, variabel kategori dapat diubah menjadi bentuk yang sesuai dengan persyaratan algoritma, sehingga memungkinkan penggunaan algoritma tersebut dalam analisis data. Dengan demikian, teknik encoding merupakan langkah penting dalam pra-pemrosesan data yang membantu meningkatkan kualitas dan kinerja model machine learning. 
2. Scaling
Setelah tahap encoding, selanjutnya adalah tahap Scaling. pada tahap ini dilakukan normalisasi skala numerical value menggunakan StandardScaler(). Teknik scaling dipilih karena pentingnya menjaga konsistensi interaksi antar variabel dalam analisis data. Dalam analisis regresi, skala yang tidak seragam antar variabel dapat menyebabkan masalah dalam menafsirkan koefisien regresi. Hal ini dikarenakan interaksi antar variabel dapat bergantung pada skala relatif dari masing-masing variabel tersebut.

    Dengan menggunakan teknik scaling, variabel-variabel dalam dataset dapat dibawa ke dalam skala yang seragam, seperti standar skala atau rentang tertentu, sehingga mengurangi potensi bias dalam analisis regresi. Teknik scaling mempengaruhi kinerja model dengan memastikan bahwa variabel-variabel memiliki dampak yang seimbang dalam model. Tanpa scaling, variabel dengan skala besar mungkin akan mendominasi pengaruh dalam analisis, sementara variabel dengan skala kecil mungkin tidak mempengaruhi model dengan signifikan. Dengan menggunakan teknik scaling, variabel-variabel dapat memiliki bobot yang seimbang dalam model, sehingga menghasilkan hasil yang lebih akurat dan dapat diandalkan.
3. Splitting Dataset
setelah data di scaling, selanjutnya splitting dataset. Splitting dataset menjadi train dan test dengan rasio 80:20, Pemisahan dataset menjadi data latih dan data uji membantu menghindari overfitting, di mana model terlalu "menghafal" data latih sehingga kinerjanya menurun saat diterapkan pada data baru. Dengan validasi menggunakan data uji yang terpisah, Evaluasi kinerja model secara lebih obyektif dan mendapatkan perkiraan yang lebih realistis tentang seberapa baik model akan berperforma pada data baru.

## Modeling
Modelling dilakukan dalam 2 tahap yaitu Find optimal model for data using TPOT, Hyperparameter Tuning untuk optimal model dan training model. berikut adalah penerapan tahap modelling:
1. Find optimal model using TPOT
TPOT merupakan AutoML yang memiliki tujuan menemukan model dengan performa tinggi terhadap dataset. pada tahap ini menggunakan TPOT Classifier karena sesuai dengan permasalahan yang ingin diselesaikan.
    hasilnya, KNN Classifier menjadi model yang optimal. Maka dari itu model machine learning yang digunakan adalah KNN Classifier.KNN (K-Nearest Neighbors) adalah salah satu algoritma machine learning yang digunakan untuk masalah klasifikasi dan regresi. Algoritma ini bekerja dengan cara menentukan label atau nilai target dari suatu data baru berdasarkan mayoritas label atau nilai target dari k data terdekat di sekitarnya, di mana k merupakan jumlah tetangga terdekat yang dipilih (biasanya merupakan bilangan ganjil untuk menghindari kebingungan jika terdapat kategori yang sama jumlahnya).
2. Hyperparameter Tuning
setelah mendapatkan model yang optimal, selanjutnya hyperparameter tuning. Hyperparameter tuning merupakan proses pencarian parameter dengan performa tinggi untuk model. Hyperparameter tuning perlu dilakukan untuk mengoptimalkan kinerja model machine learning. Hyperparameter adalah parameter yang nilainya tidak ditentukan oleh model itu sendiri, melainkan harus diatur sebelum proses pelatihan model dimulai. Contohnya adalah nilai alpha dalam regularisasi Lasso dan Ridge, jumlah pohon dalam algoritma Random Forest, atau learning rate dalam algoritma gradient boosting.

    Hyperparameter tuning dilakukan untuk mencari kombinasi nilai hyperparameter yang menghasilkan model dengan performa terbaik. Dengan melakukan tuning, kita dapat meningkatkan akurasi, generalisasi, dan kestabilan model. Tanpa tuning, model mungkin tidak akan mencapai performa optimalnya, dan bisa jadi akan overfitting atau underfitting pada data pelatihan. Metode hyperparameter tuning yang digunakan adalah GridSearchCV. GridSearchCV adalah sebuah metode yang sangat berguna dalam pengembangan model machine learning karena memungkinkan untuk mencari kombinasi hyperparameter yang optimal dengan cara yang efisien dan sistematis. Dalam praktiknya, GridSearchCV bekerja dengan mengevaluasi setiap kombinasi hyperparameter yang mungkin dari sebuah grid yang telah ditentukan sebelumnya. Setiap kombinasi dievaluasi menggunakan teknik validasi silang untuk menghindari overfitting dan underfitting.
    
    Penggunaan GridSearchCV sangat disarankan karena memungkinkan untuk menghemat waktu dan upaya yang diperlukan dalam mencari kombinasi hyperparameter yang optimal secara manual. Selain itu, GridSearchCV juga membantu dalam meningkatkan performa model dengan memilih kombinasi hyperparameter yang memberikan performa terbaik berdasarkan metrik evaluasi yang telah ditentukan. Dengan demikian, GridSearchCV merupakan salah satu alat yang sangat berguna dalam proses tuning hyperparameter untuk menghasilkan model machine learning yang lebih baik dan lebih akurat.KNN Classifier akan di tuning dengan parameter berikut:
    ```
    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]
    weights = ['uniform','distance']
    metric = ['minkowski','euclidean','manhattan']
    ```
    Hasilnya, parameter yang optimal terhadap model adalah sebagai berikut
    ```
    {'algorithm': 'auto',
     'leaf_size': 1,
     'metric': 'minkowski',
     'metric_params': None,
     'n_jobs': None,
     'n_neighbors': 5,
     'p': 1,
     'weights': 'distance'
    }
    ```
3. Training Model
setelah hyperparameter tuning, selanjutnya training model. model dengan parameter optimal dilakukan training dengan data training.
## Evaluation
setelah modeeling dilakukan, selanjutnya adalah evaluation model. pada tahap ini dilakukan evaluasi pada kinerja model menggunakan accuracy, precision, recall dan F1 score.berikut penjelasan dari masing-masing metriks:
1. Accuracy
Akurasi adalah metrik evaluasi yang mengukur seberapa baik model membuat prediksi yang benar dari total prediksi yang dilakukan. Dalam konteks klasifikasi, akurasi memberikan gambaran mengenai seberapa sering model memprediksi kelas yang benar, baik itu kelas positif maupun negatif. Sebagai panduan umum, tingkat akurasi di atas 90% sering dianggap sebagai tingkat yang baik, hal tersebut berarti model mampu melakukan prediksi dengan tingkat keberhasilan yang tinggi dalam mengklasifikasikan data ke dalam kelas yang benar.. berikut formula dari accuracy:
$$Accuracy = {TP+TN \over TP+TN+FP+FN}$$
Keterangan:
TP = True Positif
TN = True Negatif
FP = False Positif
FN = False Negatif
2. Precision
Presisi adalah metrik evaluasi yang mengukur seberapa baik model membuat prediksi yang benar untuk kelas positif dari total prediksi positif yang dilakukan. Dalam konteks klasifikasi, presisi memberikan gambaran mengenai seberapa sering model memprediksi kelas positif dengan benar, di antara semua prediksi positif yang dibuat oleh model. Tingkat precision yang tinggi menunjukkan bahwa model cenderung tidak melakukan banyak kesalahan dalam mengklasifikasikan data negatif sebagai positif (false positive). Dalam beberapa kasus, tingkat precision yang tinggi dapat dianggap sebagai indikasi kinerja model yang baik, terutama jika kesalahan prediksi positif lebih berdampak daripada kesalahan prediksi negatif. berikut adalah formula precision: 
$$Precision = {TP \over TP + FP}$$
Keterangan:
TP = True Positif
FP = False Positif
3. Recall
Recall adalah metrik evaluasi yang menggambarkan seberapa baik suatu model dalam mengidentifikasi kelas positif dengan benar. Dalam konteks klasifikasi, recall memberikan gambaran tentang seberapa baik model dalam menemukan semua kasus positif yang ada. Tingkat recall yang tinggi menunjukkan bahwa model cenderung tidak melewatkan banyak kasus positif yang sebenarnya (false negative). Dalam beberapa kasus, tingkat recall yang tinggi lebih diutamakan daripada tingkat precision yang tinggi, terutama jika kesalahan mengidentifikasi kasus positif lebih berdampak daripada kesalahan mengidentifikasi kasus negatif. berikut merupakan formula dari recall:
$$Recall = {TP \over TP + FN}$$
Keterangan:
TP = True Positif
FN = False Negatif
4. F1 Score
F1 Score merupakan metrik evaluasi yang mencerminkan keseimbangan antara Presisi (Precision) dan Sensitivitas (Recall). F1-score memberikan gambaran yang lebih komprehensif tentang kinerja model dalam mengklasifikasikan data, karena menggabungkan kedua aspek ini menjadi satu metrik tunggal. F1-score berguna terutama dalam kasus di mana kita perlu menemukan keseimbangan antara presisi dan recall. Misalnya, jika kita memiliki dataset yang tidak seimbang (imbalance classes), di mana salah satu kelas memiliki jumlah sampel yang jauh lebih sedikit dari kelas lainnya, maka F1-score dapat memberikan gambaran yang lebih baik tentang kinerja model daripada akurasi. F1-score dianggap baik jika nilainya mendekati 1, yang menunjukkan presisi dan recall yang tinggi. Namun, seperti metrik evaluasi lainnya, penilaian tentang seberapa tinggi F1-score yang dianggap "baik" juga tergantung pada konteks masalah yang sedang dihadapi. Evaluasi yang komprehensif dengan mempertimbangkan akurasi, presisi, recall, dan F1-score adalah penting untuk memahami kinerja model secara menyeluruh. 
$$F1 Score = {TP \times TN  \over TP + TN}$$
Keterangan:
TP = Recall
TN = Precision

Hasilnya, beberapa metriks evaluasi sebagai berikut: 
![evaluation matrix](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/evaluation_metrics.png?raw=true)

Gambar 10. Before Hypeparameter tuning

![evaluation matrix after gridsearchcv](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/evaluation_metrics_grid.png?raw=true)

Gambar 11. After Hypeparameter tuning

## Conclusion
Penelitian ini telah berhasil menjawab semua permasalahan yang diajukan. Pertama, penelitian ini memberikan solusi tentang cara melakukan pra-pemrosesan pada data sloth yang akan digunakan untuk membuat model yang baik. Tahapan pra-pemrosesan data meliputi: pemeriksaan dan penanganan nilai yang hilang (missing value), deteksi dan penanganan nilai-nilai yang ekstrem (outliers), pemeriksaan dan penanganan distribusi data, transformasi nilai kategorikal menjadi nilai numerik, penggunaan Standard Scaler, dan pembagian dataset menjadi data pelatihan (train) dan data pengujian (test). Kedua, dari berbagai fitur yang tersedia, ditemukan bahwa fitur yang paling berpengaruh terhadap klasifikasi sloth adalah panjang ekor (tail_length_cm) dengan korelasi negatif sebesar -0.86. Selain itu, ukuran (size_cm) dan berat (weight_kg) juga memiliki korelasi positif yang signifikan, sedangkan panjang cakar (claw_length_cm) memiliki korelasi yang lebih normal. Ketiga, penelitian ini menunjukkan bahwa proses pemodelan dilakukan dengan hasil performa yang unggul. Hasil tersebut dicapai melalui dua tahapan utama, yaitu menemukan model optimal menggunakan TPOT dan melakukan penyetelan hyperparameter. Penelitian ini menghasilkan model machine learning dengan menggunakan algoritma KNN yang mencapai akurasi sebesar 95%, presisi 100%, recall 92%, dan F1 Score 96%.

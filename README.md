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
- Bagaimana melakukan pra-pemrosesan pada data sloth yang akan digunakan untuk membuat model yang baik? 
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap klasifikasi sloth?
- Bagaimana modelling dengan hasil performa unggul?

### Goals

Berdasarkan rumusan masalah dari penelitian ini, tujuan pada penelitian ini sebagai berikut:
- Mengetahui tahapan pra pemrosesan data yang tepat dalam klasifikasi sloth
- Mengetahui fitur yang paling berkorelasi dengan klasifikasi sloth
- Membuat model machine learning yang dapat klasifikasi sloth dengan tingkat akurasi yang tinggi

 ### Solution statements
 Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
 - Untuk data understanding dan data preparation dapat dilakukan beberapa teknik, diantaranya :
    * Melakukan check dan penanganan missing value
    * Melakukan check outliers value dan melakukan penanganan outliers
    * Melakukan check & handle distribution data
    * Encoding _Categorical value into numerical value_
    * Melakukan _Standard Scaler_.
    * Melakukan pembagian dataset menjadi dua bagian dengan rasio 70% untuk data latih dan 30% untuk data uji.
* TPOT Classifier digunakan untuk mendapatkan algoritma dengan tingkat akurasi tinggi
* Hyperparameter Tuning dilakukan untuk mendapatkan parameter terbaik dari TPOT Classifier sehingga mendapatkan hasil terbaik

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
berdasarkan gambar diatas, tidak ditemukan missing value disetiap kolom. selain pengecekan dengan missing value, perlu dilakukan pengecekan terhadap invalid data, penelitian ini menggunakan kondisional sebagai berikut
```
sloth_invalid_data = (train_df["claw_length_cm"] <= 0) | (train_df["size_cm"] <= 0) | (train_df["tail_length_cm"] <= 0) | (train_df["weight_kg"] <= 0)
```
dengan kondisi filtering diatas, memiliki hasil sebagai berikut:
![grafik invalid data](https://github.com/muhammadarl/clasification-of-sloth/blob/main/src/img/invalid_data.png?raw=true)

Gambar 1. Invalid data

435 invalid data pada dataset di _drop_, berikut proses dropping invalid data:
```
rows_to_drop = train_df[sloth_invalid_data].index
train_df.drop(rows_to_drop, inplace=True)
```
### Check & handle outliers
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 2. Percentage of outliers
berdasarkan gambar diatas, terdapat outliers di beberapa kolom. 25% size_cm, 0.5 weight_kg & 0.25 claw_length_cm. outliers pada size_cm melebihi 10% sehingga data di drop dengan method IQR. setelah 1072 outliers di drop, berikut hasil setelah penanganan outliers
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 3. Grafik Sex
### Distribusi variabel
Distribusi data setiap variable terbagi 2 kelompok, yaitu categorical feature dan numerical feature.
#### Numerical feature
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 4. Grafik Sex
berdasarkan gambar 4, setiap feature terdistribusi normal dengan persebaran data berada disekitar median. jika dilihat dalam bentuk _score skewness_ setiap feature memiliki hasil sebagai berikut:
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 5. Grafik Sex
#### Categorical Feature
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 6. Grafik Sex
berdasarkan gambar 6, setiap feature tidak terdistribusi normal dengan adanya value yang mendominasi dan tidak tersebar data. maka dari itu, semua categorical feature akan di drop
### EDA-Univariate
Berikut _univariate analysis_ terhadap _categorical feature_ dan _numerical feature_
#### Categorical Feature
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 7. Grafik Sex
#### Numerical Feature
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 8. Grafik Sex
Berdasarkan gambar 8, persebaran data setiap feature terdistribusi secara normal dengan keberadaan data di sekitar nilai median.
### EDA-Multivariate
Berikut merupakan *EDA-Multivariate Analysis*:
#### Categorical Feature
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 9. Grafik Sex
Berdasarkan gambar 9, korelasi endangered dengan specie dominan pada satu value yaitu least_concern. korelasi sub specie dengan specie dominan terhadap 2 value yaitu Hoftman two toed dan slothLinnaeus two toed sloth. kedua feature tersebut di drop karena tidak berkorelasi dengan specie.
#### Numerical Feature
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 10. Grafik Sex
Berdasarkan gambar 10, korelasi antara size_cm dan tail length_cm negatif, size_cm dan weight_kg positif, tail_length_cm dan weight_kg memiliki korelasi negatif dan korelasi antara feature yang lain netral, berikut korelasi feature dengan target(specie)
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 11. Grafik Sex
Hasilnya, size_cm dan weight_kg memiliki korelasi positif dengan specie, dan tail_length_cm memiliki korelasi negatif.
## Data Preparation
Data preparation memiliki tahapan sebagai berikut encoding, Dimension reduction, Scaling, splitting dataset. Berikut penerapan dan hasil dari data preparation:
1. Encoding
Pada tahap ini, encoding dilakukan untuk 1 feature yaitu specie. penerapan encoding menggunakan label encoder karena feature ini merupakan variabel target. Hasilnya, Three toed menjadi 0 dan two toed menjadi 1. Encoding dilakukan untuk menghindari bias dan peningkatan kinerja. berikut bagaimana cara melakukan encoding.
    ```
    encoder = LabelEncoder()
    train_df["specie"] = encoder.fit_transform(train_df["specie"])
    ```
2. Scaling
Setelah tahap encoding, selanjutnya adalah tahap Scaling. pada tahap ini dilakukan normalisasi skala numerical value menggunakan StandardScaler(). tujuan scaling yaitu menjaga konsistensi interaksi antar variabel. Skala yang tidak seragam dapat menyebabkan masalah dalam menafsirkan koefisien regresi karena interaksi antar variabel dapat bergantung pada skala relatif dari variabel tersebut. berikut scaling dilakukan:
    ```
    scaler = StandardScaler()
    scaler.fit(X[numerical_feats])
    X[numerical_feats] = scaler.transform(X.loc[:, numerical_feats])
    X[numerical_feats].head()
    ```
3. Splitting Dataset
setelah data di scaling, selanjutnya splitting dataset. Splitting dataset menjadi train dan test dengan rasio 80:20, hal ini perlu dilakukan karena penting untuk menguji kinerja model pada data yang tidak digunakan dalam proses pelatihan. berikut cara splitting dataset:
    ```
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20, random_state = 1)
    ```

## Modeling
Modelling dilakukan dalam 2 tahap yaitu Find optimal model for data using TPOT, Hyperparameter Tuning untuk optimal model dan training model. TPOT merupakan AutoML yang memiliki tujuan menemukan model dengan performa tinggi terhadap dataset dan Hyperparameter tuning merupakan proses pencarian parameter dengan performa tinggi untuk model. berikut adalah penerapan tahap modelling:
1. Find optimal model using TPOT
pada tahap ini menggunakan TPOT Classifier karena sesuai dengan permasalahan yang ingin diselesaikan. berikut bagaimana menerapkan TPOT Classifier:
    ```
    # define model evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search
    model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
    # perform the search
    model.fit(X_train, y_train)
    # export the best model
    model.export('tpot_sonar_best_model.py')
    ```
    hasilnya, KNN Classifier menjadi model yang optimal. Maka dari itu model machine learning yang digunakan adalah KNN Classifier.KNN (K-Nearest Neighbors) adalah salah satu algoritma machine learning yang digunakan untuk masalah klasifikasi dan regresi. Algoritma ini bekerja dengan cara menentukan label atau nilai target dari suatu data baru berdasarkan mayoritas label atau nilai target dari k data terdekat di sekitarnya, di mana k merupakan jumlah tetangga terdekat yang dipilih (biasanya merupakan bilangan ganjil untuk menghindari kebingungan jika terdapat kategori yang sama jumlahnya).
2. Hyperparameter Tuning
setelah mendapatkan model yang optimal, selanjutnya hyperparameter tuning. KNN Classifier akan di tuning dengan parameter berikut:
    ```
    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]
    weights = ['uniform','distance']
    metric = ['minkowski','euclidean','manhattan']
    ```
    lalu, berikut bagaimana cara dilakukannya hyperparameter tuning:
    ```
    #Create new KNN object
    knn_2 = KNeighborsClassifier()
    #Use GridSearch
    clf = GridSearchCV(knn_2, grid_params, cv=cv, verbose=2)
    #Fit the model
    best_model = clf.fit(X_train,y_train)
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
setelah hyperparameter tuning, selanjutnya training model. model dengan parameter optimal dilakukan training dengan data training. berikut bagaimana training model dilakukan:
    ```
    best_model = clf.fit(X_train,y_train)
    ```
## Evaluation
setelah modeeling dilakukan, selanjutnya adalah evaluation model. pada tahap ini dilakukan evaluasi pada kinerja model menggunakan accuracy, precision, recall dan F1 score.berikut penjelasan dari masing-masing metriks:
1. Accuracy
Akurasi adalah metrik evaluasi yang mengukur seberapa baik model membuat prediksi yang benar dari total prediksi yang dilakukan. Dalam konteks klasifikasi, akurasi memberikan gambaran mengenai seberapa sering model memprediksi kelas yang benar, baik itu kelas positif maupun negatif. berikut formula dari accuracy:
$$Accuracy = {TP+TN \over TP+TN+FP+FN}$$
Keterangan:
TP = True Positif
TN = True Negatif
FP = False Positif
FN = False Negatif
2. Precision
Presisi adalah metrik evaluasi yang mengukur seberapa baik model membuat prediksi yang benar untuk kelas positif dari total prediksi positif yang dilakukan. Dalam konteks klasifikasi, presisi memberikan gambaran mengenai seberapa sering model memprediksi kelas positif dengan benar, di antara semua prediksi positif yang dibuat oleh model. berikut adalah folume precision: 
$$Precision = {TP \over TP + FP}$$
Keterangan:
TP = True Positif
FP = False Positif
3. Recall
Recall adalah metrik evaluasi yang menggambarkan seberapa baik suatu model dalam mengidentifikasi kelas positif dengan benar. berikut merupakan formula dari recall:
$$Recall = {TP \over TP + FN}$$
Keterangan:
TP = True Positif
FN = False Negatif
4. F1 Score
F1 Score merupakan metrik evaluasi yang mencerminkan keseimbangan antara Presisi (Precision) dan Sensitivitas (Recall). 
$$F1 Score = {TP \times TN  \over TP + TN}$$
Keterangan:
TP = Recall
TN = Precision

Hasilnya, beberapa metriks evaluasi sebagai berikut: 
![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 12. Before Hypeparameter tuning

![grafik sex](https://user-images.githubusercontent.com/111114060/192172057-b4ef4461-95f2-4cc7-9ba7-103b17bef14f.png)
Gambar 13. After Hypeparameter tuning
## Conclusion
Penelitian ini telah berhasil menjawab semua permasalahan yang diajukan. Pertama, penelitian ini memberikan solusi tentang cara melakukan pra-pemrosesan pada data sloth yang akan digunakan untuk membuat model yang baik. Tahapan pra-pemrosesan data meliputi: pemeriksaan dan penanganan nilai yang hilang (missing value), deteksi dan penanganan nilai-nilai yang ekstrem (outliers), pemeriksaan dan penanganan distribusi data, transformasi nilai kategorikal menjadi nilai numerik, penggunaan Standard Scaler, dan pembagian dataset menjadi data pelatihan (train) dan data pengujian (test). Kedua, dari berbagai fitur yang tersedia, ditemukan bahwa fitur yang paling berpengaruh terhadap klasifikasi sloth adalah panjang ekor (tail_length_cm) dengan korelasi negatif sebesar -0.86. Selain itu, ukuran (size_cm) dan berat (weight_kg) juga memiliki korelasi positif yang signifikan, sedangkan panjang cakar (claw_length_cm) memiliki korelasi yang lebih normal. Ketiga, penelitian ini menunjukkan bahwa proses pemodelan dilakukan dengan hasil performa yang unggul. Hasil tersebut dicapai melalui dua tahapan utama, yaitu menemukan model optimal menggunakan TPOT dan melakukan penyetelan hyperparameter. Penelitian ini menghasilkan model machine learning dengan menggunakan algoritma KNN yang mencapai akurasi sebesar 95%, presisi 100%, recall 92%, dan F1 Score 96%.

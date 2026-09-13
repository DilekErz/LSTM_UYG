# LSTM ile Türkçe Duygu Analizi

Bu proje, yazılım geliştirme stajım kapsamında **Long Short-Term Memory (LSTM)** yapısını öğrenmek ve uygulamak amacıyla geliştirilmiştir.

Projede Türkçe film yorumlarından oluşan küçük bir veri seti kullanılarak metinlerin **pozitif** veya **negatif** olarak sınıflandırılması amaçlanmıştır. Çalışma sırasında metinlerin sayısal verilere dönüştürülmesi, LSTM modelinin oluşturulması, eğitilmesi ve yeni cümleler üzerinde tahmin yapılması uygulanmıştır.

## Projenin Amacı

Bu çalışmanın temel amacı, **LSTM ağlarının sıralı veriler ve doğal dil işleme problemlerinde nasıl kullanılabileceğini** uygulamalı olarak incelemektir.

Model iki farklı sınıf üzerinde çalışmaktadır:

- `0` → Negatif
- `1` → Pozitif

Özellikle cümlenin başlangıcındaki ve sonundaki duygu değişimlerinin model tarafından öğrenilmesi amaçlanmıştır.

Örneğin:

- **"Film başlangıçta çok güzeldi ama sonunda hiç beğenmedim."** → Negatif
- **"Film başlangıçta kötüydü fakat sonunda gerçekten çok sevdim."** → Pozitif

## Kullanılan Teknolojiler

- Python
- NumPy
- TensorFlow
- Keras
- LSTM
- Natural Language Processing (NLP)

## Veri Seti

Uygulamada eğitim amacıyla Türkçe film yorumlarından oluşan küçük bir veri seti hazırlanmıştır.

Veri setinde toplam **12 cümle** bulunmaktadır:

- 6 negatif yorum
- 6 pozitif yorum

Negatif yorumlar `0`, pozitif yorumlar ise `1` etiketi ile temsil edilmiştir.

```python
labels = np.array([
    0,0,0,0,0,0,
    1,1,1,1,1,1
])
```

## Metinlerin Hazırlanması

Sinir ağları doğrudan metinlerle çalışamadığı için cümlelerin önce sayısal verilere dönüştürülmesi gerekmektedir.

Bu işlem için Keras içerisindeki `Tokenizer` kullanılmıştır.

```python
tokenizer = Tokenizer(
    num_words=200,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(sentences)
```

Burada `num_words=200` ile kullanılacak kelime sayısı sınırlandırılmıştır.

`<OOV>` ise eğitim sırasında kelime sözlüğünde bulunmayan kelimelerin temsil edilmesi için kullanılmıştır.

Daha sonra cümleler sayısal dizilere dönüştürülmüştür:

```python
sequences = tokenizer.texts_to_sequences(sentences)
```

Cümlelerin uzunlukları birbirinden farklı olduğu için `pad_sequences()` kullanılarak tüm girişlerin uzunlukları eşitlenmiştir.

```python
padded_sequences = pad_sequences(
    sequences,
    padding="post"
)
```

`padding="post"` kullanıldığı için eksik olan değerler dizilerin sonuna eklenmektedir.

En uzun cümlenin uzunluğu ise aşağıdaki şekilde elde edilmiştir:

```python
max_length = padded_sequences.shape[1]

print("En uzun cümle uzunluğu:", max_length)
```

## LSTM Modeli

Projede oluşturulan model temel olarak üç katmandan oluşmaktadır:

1. Embedding
2. LSTM
3. Dense

Model:

```python
model_lstm = Sequential([
    Embedding(
        input_dim=200,
        output_dim=16,
        mask_zero=True
    ),

    LSTM(32),

    Dense(1, activation="sigmoid")
])
```

### Embedding Katmanı

Embedding katmanı, kelimelerin sinir ağının işleyebileceği sayısal vektörler şeklinde temsil edilmesini sağlar.

```python
Embedding(
    input_dim=200,
    output_dim=16,
    mask_zero=True
)
```

Bu projede her kelime **16 boyutlu bir vektör** ile temsil edilmektedir.

`mask_zero=True` kullanılarak padding sırasında eklenen `0` değerlerinin model tarafından gerçek bir kelime gibi değerlendirilmemesi sağlanmıştır.

### LSTM Katmanı

Modelin temel katmanı:

```python
LSTM(32)
```

şeklinde oluşturulmuştur.

LSTM, klasik RNN yapısının geliştirilmiş bir türüdür ve geçmiş bilgilerin daha uzun süre korunmasını sağlayan bir hafıza mekanizmasına sahiptir.

LSTM içerisinde temel olarak şu yapılar bulunmaktadır:

- **Forget Gate:** Geçmiş bilgiden hangilerinin unutulacağını belirler.
- **Input Gate:** Yeni bilginin ne kadarının hafızaya alınacağını belirler.
- **Candidate:** Hafızaya eklenebilecek yeni bilgiyi oluşturur.
- **Output Gate:** Hangi bilginin çıktı olarak aktarılacağını belirler.

Bu mekanizmalar sayesinde LSTM, cümle içerisindeki önceki kelimelerden gelen bilgileri değerlendirebilir.

Örneğin:

> Film başlangıçta çok güzeldi **ama** sonunda hiç beğenmedim.

gibi bir cümlede yalnızca ilk kelimelere bakmak yerine cümlenin ilerleyen bölümlerindeki bilgilerin de değerlendirilmesi amaçlanmaktadır.

## LSTM Parametreleri

LSTM, SimpleRNN'e göre daha gelişmiş bir hafıza mekanizmasına sahiptir.

SimpleRNN daha basit bir hesaplama yaparken LSTM içerisinde dört farklı temel hesaplama mekanizması bulunmaktadır:

- Forget Gate
- Input Gate
- Candidate
- Output Gate

Bu nedenle LSTM katmanındaki parametre sayısı SimpleRNN katmanına göre daha fazla olabilmektedir.

Modelin katmanları ve parametre sayıları:

```python
model_lstm.summary()
```

komutu kullanılarak görüntülenmiştir.

## Dense Katmanı

Modelin son katmanında:

```python
Dense(1, activation="sigmoid")
```

kullanılmıştır.

Bu proje iki sınıflı bir sınıflandırma problemi olduğu için tek bir çıktı nöronu kullanılmıştır.

Sigmoid aktivasyon fonksiyonu modelin `0` ile `1` arasında bir tahmin değeri üretmesini sağlar.

Tahmin değerleri şu şekilde yorumlanmaktadır:

- Tahmin `< 0.5` → **NEGATİF**
- Tahmin `>= 0.5` → **POZİTİF**

## Modelin Derlenmesi

Model aşağıdaki şekilde derlenmiştir:

```python
model_lstm.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

Burada:

- `binary_crossentropy` → İkili sınıflandırma probleminde modelin hatasını hesaplar.
- `adam` → Model ağırlıklarını hataya göre güncelleyen optimizasyon algoritmasıdır.
- `accuracy` → Modelin doğru sınıflandırma oranını takip etmek için kullanılmıştır.

## Modelin Eğitilmesi

Model hazırlanan Türkçe film yorumları üzerinde **100 epoch** boyunca eğitilmiştir.

```python
model_lstm.fit(
    padded_sequences,
    labels,
    epochs=100,
    verbose=1
)
```

Eğitim sırasında model, cümleler ile bunlara karşılık gelen pozitif ve negatif etiketler arasındaki ilişkiyi öğrenmeye çalışmaktadır.

## Test Aşaması

Model eğitildikten sonra yeni cümleler üzerinde tahmin işlemi gerçekleştirilmiştir.

Kullanılan test cümleleri:

```python
test_cumleleri = [
    "Film başlangıçta çok güzeldi ama ilerleyen bölümlerde sıkıcılaştı ve sonunda hiç beğenmedim",

    "Film başlangıçta çok kötüydü fakat zamanla güzelleşti ve sonunda gerçekten çok sevdim",

    "İlk sahneler başarılıydı ama finali berbattı ve filmi sevmedim",

    "Başlangıçta sıkıcıydı fakat finali harikaydı ve filmi çok beğendim"
]
```

Her test cümlesi önce eğitim verilerinde kullanılan tokenizer ile sayısal diziye dönüştürülmektedir:

```python
sequence = tokenizer.texts_to_sequences([cumle])
```

Daha sonra eğitim verileriyle aynı uzunluğa getirilmiştir:

```python
padded = pad_sequences(
    sequence,
    maxlen=max_length,
    padding="post"
)
```

Model tahmini:

```python
tahmin = model_lstm.predict(padded, verbose=0)[0][0]
```

ile alınmıştır.

Son olarak tahmin değeri `0.5` eşik değerine göre sınıflandırılmıştır:

```python
if tahmin >= 0.5:
    print("Sonuç: POZİTİF")
else:
    print("Sonuç: NEGATİF")
```

## Örnek Tahmin Mantığı

Model her test cümlesi için `0` ile `1` arasında bir değer üretmektedir.

Örneğin:

```text
Cümle: Film başlangıçta çok güzeldi ama ilerleyen bölümlerde sıkıcılaştı ve sonunda hiç beğenmedim
Tahmin değeri: ...
Sonuç: NEGATİF
```

veya:

```text
Cümle: Film başlangıçta çok kötüydü fakat zamanla güzelleşti ve sonunda gerçekten çok sevdim
Tahmin değeri: ...
Sonuç: POZİTİF
```

Tahmin değerleri model her yeniden eğitildiğinde değişebileceği için README içerisinde sabit bir tahmin değeri verilmemiştir.

## LSTM ve SimpleRNN Arasındaki Temel Fark

Staj çalışması kapsamında daha önce incelenen **SimpleRNN** yapısına kıyasla LSTM daha gelişmiş bir hafıza mekanizmasına sahiptir.

SimpleRNN geçmiş bilgileri hidden state üzerinden taşırken, LSTM hangi bilgilerin tutulacağını, unutulacağını veya yeni hafızaya ekleneceğini kontrol eden gate mekanizmalarını kullanır.

Kısaca:

**SimpleRNN:**

```text
Geçmiş bilgi → Yeni bilgi → Hidden State → Çıktı
```

**LSTM:**

```text
Geçmiş bilgi
     ↓
Forget Gate
     ↓
Input Gate
     ↓
Candidate
     ↓
Cell State
     ↓
Output Gate
     ↓
Çıktı
```

Bu nedenle LSTM özellikle sıralı verilerde ve metinlerde geçmiş bilgilerin korunması açısından avantaj sağlayabilir.

## Proje Yapısı

```text
LSTM_UYG/
│
├── LSTM_UYG.py
└── README.md
```

## Kurulum

Projeyi çalıştırabilmek için Python'un kurulu olması gerekmektedir.

Gerekli kütüphaneler aşağıdaki komut ile kurulabilir:

```bash
pip install numpy tensorflow
```

## Çalıştırma

Proje klasöründe terminal açıldıktan sonra:

```bash
python LSTM_UYG.py
```

komutu çalıştırılabilir.

Program çalıştırıldığında:

- En uzun cümle uzunluğu
- Model özeti
- Eğitim süreci
- Test cümleleri
- Tahmin değerleri
- Pozitif veya negatif sınıflandırma sonuçları

terminal ekranında görüntülenmektedir.

## Staj Kapsamında Öğrenilen Konular

Bu uygulama ile birlikte aşağıdaki konular uygulamalı olarak incelenmiştir:

- LSTM yapısının çalışma mantığı
- RNN ve LSTM arasındaki temel farklar
- Doğal Dil İşleme (NLP)
- Türkçe metinlerle duygu analizi
- Tokenization
- Kelimelerin sayısal dizilere dönüştürülmesi
- Sequence oluşturma
- Padding işlemi
- Embedding katmanı
- LSTM katmanı
- LSTM gate mekanizmaları
- Sigmoid aktivasyon fonksiyonu
- Binary Classification
- Binary Crossentropy
- Adam optimizer
- Model eğitimi
- Yeni metinler üzerinde tahmin yapma

## Sonuç

Bu çalışma sonucunda **Long Short-Term Memory (LSTM)** mimarisinin doğal dil işleme ve duygu analizi problemlerinde nasıl kullanılabileceği uygulamalı olarak incelenmiştir.

Türkçe film yorumlarından oluşan küçük bir veri seti hazırlanmış, metinler Tokenizer kullanılarak sayısal verilere dönüştürülmüş ve LSTM tabanlı bir sinir ağı modeli oluşturulmuştur.

Model eğitildikten sonra yeni Türkçe film yorumları üzerinde tahmin işlemleri gerçekleştirilmiş ve cümleler **pozitif** veya **negatif** olarak sınıflandırılmıştır.

Bu uygulama, yazılım geliştirme stajım kapsamında gerçekleştirdiğim **yapay sinir ağları, derin öğrenme, RNN ve LSTM çalışmalarının** bir parçasıdır.

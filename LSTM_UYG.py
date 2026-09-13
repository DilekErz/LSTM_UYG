# DİLEKERSÖZ NOTE:
# LSTM UYGULAMASI

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
     
    "Filmin başlangıcı çok güzeldi oyunculukları da beğendim ama ilerleyen sahnelerde hikaye çok sıkıcı oldu ve sonunda hiç sevmedim",
    "İlk başta film bana güzel geldi fakat zaman geçtikçe senaryo kötüleşti ve final beni çok hayal kırıklığına uğrattı",
    "Oyuncular başarılıydı ve başlangıç umut vericiydi ancak hikaye giderek kötüleşti ve filmi beğenmedim",
    "Filmin ilk yarısı güzel olsa da ikinci yarısı çok sıkıcıydı ve sonuç olarak filmi sevmedim",
    "Başlangıçta çok eğlendim fakat filmin sonlarına doğru hikaye anlamsızlaştı ve sonunda hiç beğenmedim",
    "İlk sahneler başarılıydı ama ilerleyen bölümlerde oyunculuklar kötüleşti ve filmden hiç hoşlanmadım",

    "Filmin başlangıcı biraz sıkıcıydı fakat ilerleyen sahnelerde hikaye çok güzelleşti ve sonunda filmi çok beğendim",
    "İlk başta film bana kötü geldi ancak zaman geçtikçe senaryo gelişti ve finali gerçekten çok sevdim",
    "Oyunculuklar başlangıçta zayıftı fakat hikaye ilerledikçe film güzelleşti ve sonunda çok beğendim",
    "Filmin ilk yarısı sıkıcı olsa da ikinci yarısı harikaydı ve sonuç olarak filmi çok sevdim",
    "Başlangıçta filmi sevmedim fakat sonlara doğru hikaye çok başarılı hale geldi ve finaline bayıldım",
    "İlk sahneler kötüydü ama ilerleyen bölümlerde oyunculuklar gelişti ve film gerçekten çok güzeldi"
]

labels = np.array([
    0,0,0,0,0,0,
    1,1,1,1,1,1
])

tokenizer = Tokenizer(
    num_words=200,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

padded_sequences = pad_sequences(
    sequences,
    padding="post"
)

max_length = padded_sequences.shape[1]

print("En uzun cümle uzunluğu:", max_length)

model_lstm = Sequential([
    Embedding(
        input_dim=200,
        output_dim=16,
        mask_zero=True
    ),

    LSTM(32),
    #  tek fark burda LSTM'de ise daha gelişmiş bir hafıza mekanizması var Yani LSTM kabaca:Geçmişten gelen hangi bilgiyi tutayım, hangisini unutayım ve yeni gelen hangi bilgiyi hafızaya ekleyeyim

    Dense(1, activation="sigmoid")
])

model_lstm.build(input_shape=(None, max_length))

model_lstm.summary()

# çıktıda lstm neden 1600 sorarak SimpleRNN daha basit bir hafıza mekanizmasına sahipti. LSTM ise bilgiyi yönetmek için 4 ayrı hesaplama mekanizması kullanıyor
# Forget gate: Geçmiş bilgiden neyi unutayım?
# Input gate: Yeni bilginin ne kadarını hafızaya ekleyeyim?
# Candidate: Hafızaya eklenebilecek yeni bilgiyi oluştur.
# Output gate: Hangi bilgiyi çıktı/hidden state olarak aktaracağım 
# 400x4 =1600 

model_lstm.compile(
    loss="binary_crossentropy", #pozitif/negatif gibi ikili sınıflandırmada hatayı hesaplıyor
    optimizer="adam", #LSTM’nin 2017 parametresini, yaptığı hataya göre güncelliyor
    metrics=["accuracy"]
)

model_lstm.fit(
    padded_sequences,
    labels,
   epochs=100,
    verbose=1
)


test_cumleleri = [
    "Film başlangıçta çok güzeldi ama ilerleyen bölümlerde sıkıcılaştı ve sonunda hiç beğenmedim",

    "Film başlangıçta çok kötüydü fakat zamanla güzelleşti ve sonunda gerçekten çok sevdim",

    "İlk sahneler başarılıydı ama finali berbattı ve filmi sevmedim",

    "Başlangıçta sıkıcıydı fakat finali harikaydı ve filmi çok beğendim"
]

for cumle in test_cumleleri:

    sequence = tokenizer.texts_to_sequences([cumle])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post"
    )

    tahmin = model_lstm.predict(padded, verbose=0)[0][0]

    print("\nCümle:", cumle)
    print("Tahmin değeri:", tahmin)

    if tahmin >= 0.5:
        print("Sonuç: POZİTİF")
    else:
        print("Sonuç: NEGATİF")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import ttk
import webbrowser

# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Benzer Kelime Bulucu")

# Veri setini yükle
df = pd.read_csv('yeni.csv')

def train_model(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['turkish'])

    model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    model.fit(vectors, data['Label'])

    return vectorizer, model

# 'turkish' sütunundaki NaN değerleri temizle
df = df.dropna(subset=['turkish'])

# 'turkish' sütununu kullan (eğer veri setiniz farklı ise uygun sütunu seçin)
word_list = df['turkish'].dropna().tolist()

# Veri setindeki kelimelerin etiketlerini belirle
df['Label'] = range(len(df))

# Modeli eğit
vectorizer, model = train_model(df)

def find_similar_words(input_word, vectorizer, model, word_list, threshold=0.5):
    input_vector = vectorizer.transform([input_word])
    predictions = model.kneighbors(input_vector, n_neighbors=len(model.classes_), return_distance=True)
    # Benzer kelimeleri bul
    similar_words = []
    for index, distance in zip(predictions[1][0], predictions[0][0]):
        if distance < threshold:  # Eşik değerinden küçük olanları ele
            similar_words.append((word_list[index], distance))
    return similar_words

def on_submit():
    input_word = entry_word.get()

    # Benzer kelimeleri bul
    similar_words = find_similar_words(input_word, vectorizer, model, word_list)

    # Sonuçları yazdır
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "\nBenzer kelimeler:\n")
    if not similar_words:
        result_text.insert(tk.END, "Benzer kelime bulunamadı.\n")
    else:
        for word, similarity in similar_words:
            result_text.insert(tk.END, f"{word} - Benzerlik: {similarity}\n")
    result_text.config(state=tk.DISABLED)

def on_exit():
    root.destroy()

def find_meaning():
    input_word = entry_word.get()
    tdk_url = f"https://sozluk.gov.tr/?kelime={input_word}"
    webbrowser.open_new(tdk_url)

# Giriş etiketi ve giriş kutusu
label_word = ttk.Label(root, text="Bir kelime girin:")
label_word.grid(row=0, column=0, padx=10, pady=10, sticky="e")

entry_word = ttk.Entry(root)
entry_word.grid(row=0, column=1, padx=10, pady=10, sticky="w")

# "Ara" butonu
submit_button = ttk.Button(root, text="Ara", command=on_submit)
submit_button.grid(row=1, column=0, columnspan=2, pady=10)

# "Anlamı Bul" butonu
meaning_button = ttk.Button(root, text="Kelimenin Anlamına Bak", command=find_meaning)
meaning_button.grid(row=1, column=2, pady=10)

# "Çıkış" butonu
exit_button = ttk.Button(root, text="Çıkış", command=on_exit)
exit_button.grid(row=1, column=3, pady=10)

# Sonuç etiketi
result_text = tk.Text(root, height=10, width=50, wrap=tk.WORD, state=tk.DISABLED)
result_text.grid(row=2, column=0, columnspan=4, pady=10)

# Pencereyi başlat
root.mainloop()

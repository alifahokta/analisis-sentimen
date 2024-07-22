from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
import pandas as pd
import os
import re
import string
import math
import time
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed
import nltk
from wordcloud import WordCloud
from collections import Counter
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'analisis'

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sentiment_analysis'

mysql = MySQL(app)

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('indonesian'))

# Menambahkan daftar kata-kata spesifik untuk dihapus
additional_stopwords = ['rb', 'nya', 'yg', 'yang', 'tpi', 'kmrn', 'dgn', 'd', 'trus', 'gk', 'udh', 'sm', 'dpt', 'tdk', 'dr', 'org', 'dn'] + list('abcdefghijklmnopqrstuvwxyz')

def preprocess_text(text):
    # Cleaning: menghilangkan angka, tanda baca dan enter
    cleaned_text = re.sub(r'\d+', '', text)
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
    cleaned_text = cleaned_text.replace('\n', '').replace('\r', '')

    # Case folding: mengubah teks menjadi huruf kecil
    case_folded_text = cleaned_text.lower()

    # Tokenizing: memisahkan teks menjadi token
    words = word_tokenize(case_folded_text)

    # Stopword removal: menghilangkan stopwords dan menghapus kata tertentu
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Remove additional stopwords and single letters
    words = [word for word in words if word not in additional_stopwords]

    # Stemming: mengembalikan kata ke bentuk dasarnya
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    return {
        'cleaning': cleaned_text,
        'case_folding': case_folded_text,
        'tokenizing': words,
        'stopword_removal': words,
        'stemming': ' '.join(stemmed_words)
    }

def preprocess_data():
    # Ambil data dari MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT text, label FROM reviews")
    data = cur.fetchall()

    # Convert data to DataFrame
    data = pd.DataFrame(data, columns=['text', 'label'])

    # Preprocessing
    start_time = time.time()
    data['preprocessed_text'] = Parallel(n_jobs=-1)(delayed(preprocess_text)(text) for text in data['text'])
    preprocess_time = time.time() - start_time

    # Simpan hasil preprocessing ke MySQL
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM preprocessed_reviews")
    for index, row in data.iterrows():
        cur.execute("INSERT INTO preprocessed_reviews (original_text, cleaned_text, case_folded_text, tokenized_text, stopword_removed_text, stemmed_text) VALUES (%s, %s, %s, %s, %s, %s)", 
                    (row['text'], row['preprocessed_text']['cleaning'], row['preprocessed_text']['case_folding'], ' '.join(row['preprocessed_text']['tokenizing']), ' '.join(row['preprocessed_text']['stopword_removal']), row['preprocessed_text']['stemming']))
    mysql.connection.commit()

    return preprocess_time

def perform_classification(split_ratio=0.3):
    # Ambil data yang sudah di-preprocessing dari MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT pr.stemmed_text, r.label FROM preprocessed_reviews pr INNER JOIN reviews r ON pr.id = r.id")
    
    data = cur.fetchall()

    # Convert data to DataFrame
    data = pd.DataFrame(data, columns=['text', 'label'])
    texts = data['text']
    y = data['label']
    
    # Split data menjadi train dan test set
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=split_ratio, random_state=42)
    
    # Menghitung Term Frequency (TF)
    start_time = time.time()
    word_counts = defaultdict(lambda: defaultdict(int))
    category_word_counts = defaultdict(int)

    for doc, category in zip(X_train, y_train):
        for word in doc.split():
            word_counts[category][word] += 1
            category_word_counts[category] += 1

    # Menghitung Prior Probability (P(Cj))
    category_counts = Counter(y_train)
    total_docs = sum(category_counts.values())
    priors = {category: count / total_docs for category, count in category_counts.items()}

    # Menghitung Likelihood (P(W|Cj)) dengan smoothing Laplace
    vocab = set(word for doc in X_train for word in doc.split())
    vocab_size = len(vocab)
    likelihoods = defaultdict(lambda: defaultdict(float))

    for category in word_counts:
        total_words_in_category = category_word_counts[category]
        for word in vocab:
            likelihoods[category][word] = (word_counts[category][word] + 1) / (total_words_in_category + vocab_size)
    training_time = time.time() - start_time

    # Membuat Vmap untuk setiap kategori
    def calculate_vmap(document):
        category_scores = {}
        for category in priors:
            score = math.log(priors[category])
            for word in document.split():
                if word in vocab:
                    score += math.log(likelihoods[category][word])
            category_scores[category] = score
        return category_scores

    # Mencari nilai maksimum dari Vmap untuk prediksi
    def predict(document):
        vmap = calculate_vmap(document)
        return max(vmap, key=vmap.get)

    # Evaluasi model pada data uji
    start_time = time.time()
    y_pred = Parallel(n_jobs=-1)(delayed(predict)(doc) for doc in X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)  # Output as dictionary
    confusion_mtx = confusion_matrix(y_test, y_pred)

    # Extract precision, recall, and f1-score
    metrics = {
    'precision': "{:.2f}%".format(classification_rep['weighted avg']['precision'] * 100),
    'recall': "{:.2f}%".format(classification_rep['weighted avg']['recall'] * 100),
    'f1-score': "{:.2f}%".format(classification_rep['weighted avg']['f1-score'] * 100)
    }
    
    # Simpan hasil prediksi ke MySQL
    cur.execute("DELETE FROM predictions")  # Clear existing data
    for original_text, original_sentiment, predicted_sentiment in zip(X_test, y_test, y_pred):
        cur.execute("INSERT INTO predictions (preprocessing_text, original_sentiment, predicted_sentiment) VALUES (%s, %s, %s)",
                    (original_text, original_sentiment, predicted_sentiment))
    mysql.connection.commit()

    return accuracy, metrics, confusion_mtx, X_test, y_pred, y_test, training_time, predict_time

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# Batasan ukuran file unggahan (misalnya, 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/train_data', methods=['GET', 'POST'])
def train_data():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Baca file CSV dan masukkan ke database
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Lewati header
                for row in reader:
                    text, label = row
                    cur = mysql.connection.cursor()
                    cur.execute("INSERT INTO reviews (text, label) VALUES (%s, %s)", (text, label))
                mysql.connection.commit()

            os.remove(file_path)
            flash('Data berhasil diunggah', 'success')
        else:
            flash('Unggah file CSV yang valid', 'danger')
        return redirect(url_for('train_data'))

    # Ambil data dari MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM reviews")
    data = cur.fetchall()
    data_dict = [{'id': row[0], 'text': row[1], 'label': row[2]} for row in data]
    cur.close()

    return render_template('train_data.html', data=data_dict)

@app.route('/update_review', methods=['POST'])
def update_review():
    id = request.form['id']
    text = request.form['text']
    label = request.form['label']

    print(f"Received: id={id}, text={text}, label={label}")  # Debugging line
    
    if id and text and label:
        cur = mysql.connection.cursor()
        try:
            cur.execute("""
                UPDATE reviews
                SET text=%s, label=%s
                WHERE id=%s
            """, (text, label, int(id)))

            cur.execute("""
                UPDATE preprocessed_reviews
                SET original_text=%s
                WHERE id=%s
            """, (text, int(id)))

            mysql.connection.commit()
            flash('Review berhasil diupdate', 'success')
        except Exception as e:
            mysql.connection.rollback()
            flash(str(e), 'danger')
        finally:
            cur.close()
    else:
        flash('Data tidak lengkap', 'danger')

    return redirect(url_for('train_data'))

@app.route('/delete_review', methods=['POST'])
def delete_review():
    id = request.form['id']
    
    if id:
        cur = mysql.connection.cursor()
        try:
            cur.execute("DELETE FROM reviews WHERE id=%s", (int(id),))
            cur.execute("DELETE FROM preprocessed_reviews WHERE id=%s", (int(id),))
            mysql.connection.commit()
            flash('Review berhasil dihapus', 'success')
        except Exception as e:
            mysql.connection.rollback()
            flash(str(e), 'danger')
        finally:
            cur.close()
    else:
        flash('ID tidak ditemukan', 'danger')

    return redirect(url_for('train_data'))

@app.route('/test_data')
def test_data():
    return render_template('test_data.html')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        # Jalankan preprocessing
        preprocess_time = preprocess_data()
        return redirect(url_for('preprocessing'))

    # Ambil data dari MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT original_text, cleaned_text, case_folded_text, tokenized_text, stopword_removed_text, stemmed_text FROM preprocessed_reviews")
    data = cur.fetchall()
    combined_data = [{'original': row[0], 'preprocessing': {
        'cleaning': row[1],
        'case_folding': row[2],
        'tokenizing': row[3].split(),
        'stopword_removal': row[4].split(),
        'stemming': row[5]
    }} for row in data]
    return render_template('preprocessing.html', combined_data=combined_data)

@app.route('/naive_bayes_classification', methods=['GET', 'POST'])
def naive_bayes_classification():
    if request.method == 'POST':
        split_ratio = request.form.get('split_ratio', 0.3)
        try:
            split_ratio = float(split_ratio)
        except ValueError:
            split_ratio = 0.3  # Default value if conversion fails

        # Perform classification dengan split ratio yang dimasukkan
        accuracy, metrics, confusion_mtx, X_test, y_pred, y_test, training_time, predict_time = perform_classification(split_ratio=split_ratio)

        # Simpan predictions ke MySQL
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM predictions")  # Clear existing data
        for original_text, original_sentiment, predicted_sentiment in zip(X_test, y_test, y_pred):
            cur.execute("INSERT INTO predictions (preprocessing_text, original_sentiment, predicted_sentiment) VALUES (%s, %s, %s)",
                        (original_text, original_sentiment, predicted_sentiment))
        mysql.connection.commit()
        
        # Tampilkan prediksi sebelumnya
        cur.execute("SELECT preprocessing_text, original_sentiment, predicted_sentiment FROM predictions")
        data = cur.fetchall()
        cur.close()

        predictions = []
        for row in data:
            prediction = {
                'preprocessing_text': row[0],
                'original_sentiment': row[1],
                'predicted_sentiment': row[2]
            }
            predictions.append(prediction)

        return render_template('naive_bayes_classification.html', accuracy=accuracy,
                               metrics=metrics, confusion_mtx=confusion_mtx,
                               training_time=training_time, predict_time=predict_time,
                               X_test=X_test, y_pred=y_pred, y_test=y_test, split_ratio=split_ratio, predictions=predictions)

    # Ambil data dari MySQL untuk ditampilkan saat GET request
    cur = mysql.connection.cursor()
    cur.execute("SELECT preprocessing_text, original_sentiment, predicted_sentiment FROM predictions")
    data = cur.fetchall()
    cur.close()

    predictions = []
    for row in data:
        prediction = {
            'preprocessing_text': row[0],
            'original_sentiment': row[1],
            'predicted_sentiment': row[2]
        }
        predictions.append(prediction)

    return render_template('naive_bayes_classification.html', predictions=predictions)

@app.route('/accuracy_testing', methods=['GET', 'POST'])
def accuracy_testing():
    if request.method == 'POST':
        split_ratio = float(request.form['split_ratio'])
        accuracy, metrics, confusion_mtx, _, _, _, training_time, predict_time = perform_classification(split_ratio=split_ratio)
        accuracy_percent = "{:.2%}".format(accuracy)  # Konversi nilai akurasi menjadi persen dengan dua desimal
        return render_template('accuracy_testing.html', accuracy=accuracy_percent,
                               metrics=metrics, confusion_mtx=confusion_mtx,
                               training_time=training_time, predict_time=predict_time, split_ratio=split_ratio)

    # Jika GET request, tampilkan halaman dengan default split_ratio 0.3 (70:30)
    accuracy, metrics, confusion_mtx, _, _, _, training_time, predict_time = perform_classification(split_ratio=0.3)
    accuracy_percent = "{:.2%}".format(accuracy)  # Konversi nilai akurasi menjadi persen dengan dua desimal
    return render_template('accuracy_testing.html', accuracy=accuracy_percent, metrics=metrics,
                           confusion_mtx=confusion_mtx, training_time=training_time, predict_time=predict_time, split_ratio=0.3)

@app.route('/sentiment_visualization')
def sentiment_visualization():
    # Ambil data sentimen dari MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT original_sentiment, predicted_sentiment, preprocessing_text FROM predictions")
    data = cur.fetchall()
    cur.close()

    # Hitung jumlah data positif, netral, dan negatif untuk original_sentiment
    original_sentiment_counts = {
        'Positive': sum(1 for row in data if row[0] == 'positif'),
        'Neutral': sum(1 for row in data if row[0] == 'netral'),
        'Negative': sum(1 for row in data if row[0] == 'negatif')
    }

    # Hitung jumlah data positif, netral, dan negatif untuk predicted_sentiment
    predicted_sentiment_counts = {
        'Positive': sum(1 for row in data if row[1] == 'positif'),
        'Neutral': sum(1 for row in data if row[1] == 'netral'),
        'Negative': sum(1 for row in data if row[1] == 'negatif')
    }

    # Memisahkan teks berdasarkan sentimen
    positive_text = " ".join([row[2] for row in data if row[1] == 'positif'])
    neutral_text = " ".join([row[2] for row in data if row[1] == 'netral'])
    negative_text = " ".join([row[2] for row in data if row[1] == 'negatif'])

    # Membuat WordCloud untuk masing-masing sentimen
    wordcloud_positive = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
    wordcloud_neutral = WordCloud(width=800, height=400, background_color="white").generate(neutral_text)
    wordcloud_negative = WordCloud(width=800, height=400, background_color="white").generate(negative_text)

    # Simpan WordCloud sebagai gambar
    wordcloud_positive.to_file('static/wordcloud_positive.png')
    wordcloud_neutral.to_file('static/wordcloud_neutral.png')
    wordcloud_negative.to_file('static/wordcloud_negative.png')
    
    # Filter stop words dari setiap kategori teks
    positive_words = [word for word in positive_text.split() if word not in stop_words]
    neutral_words = [word for word in neutral_text.split() if word not in stop_words]
    negative_words = [word for word in negative_text.split() if word not in stop_words]

    # Hitung frekuensi kata
    positive_word_counts = Counter(positive_words)
    neutral_word_counts = Counter(neutral_words)
    negative_word_counts = Counter(negative_words)

    # Mengambil 5 kata yang paling banyak muncul
    positive_most_common = positive_word_counts.most_common(5)
    neutral_most_common = neutral_word_counts.most_common(5)
    negative_most_common = negative_word_counts.most_common(5)

    # Hitung jumlah kata pada setiap sentimen
    positive_word_count = len(positive_words)
    neutral_word_count = len(neutral_words)
    negative_word_count = len(negative_words)

    return render_template('sentiment_visualization.html',
                           original_sentiment_counts=original_sentiment_counts,
                            predicted_sentiment_counts=predicted_sentiment_counts,
                           positive_most_common=positive_most_common,
                           neutral_most_common=neutral_most_common,
                           negative_most_common=negative_most_common,
                           positive_word_count=positive_word_count,
                           neutral_word_count=neutral_word_count,
                           negative_word_count=negative_word_count)

if __name__ == '__main__':
    app.run(debug=True, port=8000)

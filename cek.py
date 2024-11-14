from flask import Flask, render_template
from flask_mysqldb import MySQL
import pandas as pd

app = Flask(__name__)
app.secret_key = 'analisis'

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sentiment_analysis'

mysql = MySQL(app)

@app.route('/')
def index():
    try:
        # Ambil data untuk analisis 6 bulan dari all_reviews
        cur = mysql.connection.cursor()
        query = "SELECT text, publishedAtDate FROM all_reviews"
        cur.execute(query)
        reviews_data = cur.fetchall()
        
        # Cek data yang diambil
        #print("Data from database:", reviews_data)

        # Convert data ke DataFrame
        df = pd.DataFrame(reviews_data, columns=['text', 'publishedAtDate'])
        df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
        df.dropna(subset=['publishedAtDate'], inplace=True)

        # Cek DataFrame setelah format
        print("DataFrame after formatting and parsing:")
        print(df.head())
    
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return "An error occurred"
    
    finally:
        # Jangan lupa untuk menutup cursor
        cur.close()
    
    # Render template atau proses lebih lanjut
    #return render_template('index.html', reviews_data=df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

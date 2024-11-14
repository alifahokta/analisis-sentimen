import mysql.connector

# Menghubungkan ke database MySQL
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",  # Ganti dengan password Anda
  database="sentiment_analysis"  # Ganti dengan nama database Anda
)

# Membuat kursor untuk melakukan query
mycursor = mydb.cursor()

# Menjalankan query untuk menghitung jumlah baris di tabel reviews
mycursor.execute("SELECT COUNT(*) FROM all_reviews")

# Mengambil hasil query
result = mycursor.fetchone()[0]

print("Jumlah data dalam tabel reviews:", result)

# Menutup koneksi
mycursor.close()
mydb.close()

import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file Excel
file_path = 'Data Curah Hujan Harian 2022-2024 Cleaned.xlsx'
df = pd.read_excel(file_path, sheet_name='Data Harian - Table', skiprows=10)

# Membersihkan data dengan mengganti nama kolom dan menghapus kolom yang tidak relevan
df.columns = ['Tanggal', 'Curah Hujan', 'Kolom_Tidak_Terpakai', 'Keterangan']
df = df.drop(columns=['Kolom_Tidak_Terpakai', 'Keterangan'])

# Mengonversi kolom 'Tanggal' ke format datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

# Menghapus baris yang memiliki tanggal NaT (tanggal tidak valid) atau curah hujan yang hilang
df = df.dropna(subset=['Tanggal', 'Curah Hujan'])

# Memastikan kolom 'Curah Hujan' bertipe numerik
df['Curah Hujan'] = pd.to_numeric(df['Curah Hujan'], errors='coerce')

# Menyimpan 'Tanggal' sebagai kolom untuk referensi
df['Tahun'] = df['Tanggal'].dt.year

# Memisahkan data berdasarkan tahun
data_per_tahun = {year: data for year, data in df.groupby('Tahun')}

# Membuat grafik linier untuk setiap tahun
plt.figure(figsize=(12, 8))
for year, data in data_per_tahun.items():
    plt.plot(data['Tanggal'], data['Curah Hujan'], label=f'Tahun {year}')

plt.title('Curah Hujan Harian per Tahun', fontsize=16, fontweight='bold')
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Curah Hujan (mm)', fontsize=12)
plt.legend(title='Tahun')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
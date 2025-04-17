import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
df['Bulan'] = df['Tanggal'].dt.month

# Menetapkan kolom 'Tanggal' sebagai indeks
df.set_index('Tanggal', inplace=True)

# Menampilkan statistik bulanan
statistik_bulanan = df.resample('M').agg(
    Rata_Rata=('Curah Hujan', 'mean'),
    Median=('Curah Hujan', 'median'),
    Std_Dev=('Curah Hujan', 'std'),
    Maksimal=('Curah Hujan', 'max')
)
print(statistik_bulanan)

# Memisahkan data berdasarkan tahun
df_2022 = df[df['Tahun'] == 2022]
df_2023 = df[df['Tahun'] == 2023]
df_2024 = df[df['Tahun'] == 2024]

# Membuat boxplot untuk curah hujan tahun 2022, dengan pengelompokan berdasarkan bulan
plt.figure(figsize=(10, 6))  # Ukuran grafik
sns.set(style="whitegrid")  # Menggunakan grid latar belakang untuk memperjelas boxplot

# Boxplot untuk tahun 2022
ax = sns.boxplot(x='Bulan', y='Curah Hujan', data=df_2022, palette="Set2", width=0.6)
plt.title('Boxplot Curah Hujan Tahun 2022', fontsize=16, fontweight='bold')
plt.xlabel('Bulan', fontsize=12)
plt.ylabel('Curah Hujan (mm)', fontsize=12)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Boxplot untuk tahun 2023
plt.figure(figsize=(10, 6))  # Ukuran grafik
ax = sns.boxplot(x='Bulan', y='Curah Hujan', data=df_2023, palette="Set2", width=0.6)
plt.title('Boxplot Curah Hujan Tahun 2023', fontsize=16, fontweight='bold')
plt.xlabel('Bulan', fontsize=12)
plt.ylabel('Curah Hujan (mm)', fontsize=12)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Boxplot untuk tahun 2024
plt.figure(figsize=(10, 6))  # Ukuran grafik
ax = sns.boxplot(x='Bulan', y='Curah Hujan', data=df_2024, palette="Set2", width=0.6)
plt.title('Boxplot Curah Hujan Tahun 2024', fontsize=16, fontweight='bold')
plt.xlabel('Bulan', fontsize=12)
plt.ylabel('Curah Hujan (mm)', fontsize=12)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

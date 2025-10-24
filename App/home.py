# ======================================================================================
# 1. IMPORT PUSTAKA UTAMA
# ======================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# ======================================================================================
# 2. KONFIGURASI HALAMAN & STATE MANAGEMENT
# ======================================================================================
st.set_page_config(
    page_title="Analisis & Deteksi Malware | Proyek Keamanan Data",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi session state untuk data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ======================================================================================
# 3. FUNGSI CACHING UNTUK MEMUAT ASET (MODEL & DATASET)
# ======================================================================================

@st.cache_resource
def load_model(model_path):
    """Memuat model XGBoost yang telah dilatih dengan penanganan error yang kuat."""
    if not os.path.exists(model_path):
        st.error(f"FATAL: File model tidak ditemukan di '{model_path}'. Pastikan path sudah benar.")
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Terjadi error kritis saat memuat model: {e}")
        st.stop()

@st.cache_data
def load_dataset(dataset_path):
    """Memuat dataset untuk EDA dengan penanganan error."""
    if not os.path.exists(dataset_path):
        st.error(f"FATAL: File dataset tidak ditemukan di '{dataset_path}'. Pastikan path sudah benar.")
        st.stop()
    try:
        df = pd.read_csv(dataset_path, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Terjadi error kritis saat memuat dataset: {e}")
        st.stop()

# ======================================================================================
# 4. FUNGSI PREPROCESSING & FEATURE ENGINEERING (DENGAN PERBAIKAN TIPE DATA)
# ======================================================================================

def preprocess_for_prediction(df_input):
    """
    Fungsi cerdas yang memproses data input, termasuk konversi tipe data,
    feature engineering, encoding, dan penyesuaian kolom.
    """
    df = df_input.copy()

    # --- LANGKAH 1: Hapus kolom ID dan Target ---
    cols_to_drop = ['md5', 'sha1', 'Class', 'Category', 'Family']
    cols_dropped = [col for col in cols_to_drop if col in df.columns]
    if cols_dropped:
        df = df.drop(columns=cols_dropped, errors='ignore')
        st.info(f"Info: Kolom berikut secara otomatis dihapus: `{', '.join(cols_dropped)}`.")

    # --- LANGKAH 2: Konversi Tipe Data (PERBAIKAN KRUSIAL) ---
    # Fungsi ini mengatasi error ValueError dengan mengubah string (object) menjadi numerik.
    def robust_to_numeric(val):
        """Fungsi untuk mengonversi nilai, termasuk string hex ('0x...'), ke numerik."""
        if isinstance(val, (int, float, np.number)):
            return val
        if not isinstance(val, str):
            return np.nan
        val = val.strip().lower()
        try:
            if val.startswith('0x'):
                return int(val, 16)
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    # Daftar kolom yang seharusnya numerik tetapi sering dibaca sebagai 'object'
    potential_numeric_cols = [
        'EntryPoint', 'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData',
        'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase',
        'SectionAlignment', 'FileAlignment', 'OperatingSystemVersion', 'ImageVersion',
        'SizeOfImage', 'SizeOfHeaders', 'Checksum', 'SizeofStackReserve',
        'SizeofStackCommit', 'SizeofHeapCommit', 'SizeofHeapReserve', 'LoaderFlags',
        'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData',
        'text_PointerToRawData', 'text_PointerToRelocations', 'text_PointerToLineNumbers',
        'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData',
        'rdata_PointerToRawData', 'rdata_PointerToRelocations', 'rdata_PointerToLineNumbers'
    ]
    
    for col in potential_numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(robust_to_numeric)
            df[col] = df[col].fillna(0) # Ganti nilai yang gagal dikonversi dengan 0

    # --- LANGKAH 3: Feature Engineering ---
    with st.expander("Lihat Proses Feature Engineering"):
        st.write("Membuat fitur-fitur baru berdasarkan data yang ada...")
        files_cols = [col for col in df.columns if 'files_' in col]
        processes_cols = [col for col in df.columns if 'processes_' in col]
        network_cols = [col for col in df.columns if 'network_' in col]
        registry_cols = [col for col in df.columns if 'registry_' in col]

        df['total_files'] = df[files_cols].sum(axis=1) if files_cols else 0
        df['total_processes'] = df[processes_cols].sum(axis=1) if processes_cols else 0
        df['ratio_malicious_files'] = df['files_malicious'] / (df['total_files'] + 1e-6) if 'files_malicious' in df.columns and files_cols else 0
        df['ratio_suspicious_processes'] = df['processes_suspicious'] / (df['total_processes'] + 1e-6) if 'processes_suspicious' in df.columns and processes_cols else 0
        df['total_network_activity'] = df[network_cols].sum(axis=1) if network_cols else 0
        df['has_network_threat'] = np.where(df['network_threats'] > 0, 1, 0) if 'network_threats' in df.columns else 0
        df['total_registry'] = df[registry_cols].sum(axis=1) if registry_cols else 0
        
        risk_components = {'files_malicious': 2, 'processes_malicious': 2, 'files_suspicious': 1, 'processes_suspicious': 1, 'network_threats': 1}
        df['risk_score'] = sum(df.get(col, 0) * weight for col, weight in risk_components.items())
        df['high_risk_flag'] = np.where(df['risk_score'] > 0.0, 1, 0)
        
        new_features = ['total_files', 'total_processes', 'ratio_malicious_files', 'ratio_suspicious_processes', 'total_network_activity', 'has_network_threat', 'total_registry', 'risk_score', 'high_risk_flag']
        st.success(f"Fitur baru berhasil dibuat: `{', '.join(new_features)}`.")

    # --- LANGKAH 4: Encoding Fitur Kategorikal ---
    # Daftar ini HANYA berisi kolom yang benar-benar kategorikal.
    cat_cols = [
        'file_extension', 'PEType', 'MachineType', 'magic_number',
        'bytes_on_last_page', 'pages_in_file', 'relocations', 'size_of_header',
        'min_extra_paragraphs', 'max_extra_paragraphs', 'init_ss_value',
        'init_sp_value', 'init_ip_value', 'init_cs_value', 'over_lay_number',
        'oem_identifier', 'address_of_ne_header', 'Magic', 'Subsystem',
        'DllCharacteristics', 'text_Characteristics', 'rdata_Characteristics'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # --- LANGKAH 5: Penyelarasan Kolom Final ---
    final_feature_order = ['file_extension', 'EntryPoint', 'PEType', 'MachineType', 'magic_number', 'bytes_on_last_page', 'pages_in_file', 'relocations', 'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs', 'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value', 'over_lay_number', 'oem_identifier', 'address_of_ne_header', 'Magic', 'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase', 'SectionAlignment', 'FileAlignment', 'OperatingSystemVersion', 'ImageVersion', 'SizeOfImage', 'SizeOfHeaders', 'Checksum', 'Subsystem', 'DllCharacteristics', 'SizeofStackReserve', 'SizeofStackCommit', 'SizeofHeapCommit', 'SizeofHeapReserve', 'LoaderFlags', 'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData', 'text_PointerToRawData', 'text_PointerToRelocations', 'text_PointerToLineNumbers', 'text_Characteristics', 'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData', 'rdata_PointerToRawData', 'rdata_PointerToRelocations', 'rdata_PointerToLineNumbers', 'rdata_Characteristics', 'registry_read', 'registry_write', 'registry_delete', 'registry_total', 'network_threats', 'network_dns', 'network_http', 'network_connections', 'processes_malicious', 'processes_suspicious', 'processes_monitored', 'total_procsses', 'files_malicious', 'files_suspicious', 'files_text', 'files_unknown', 'dlls_calls', 'apis', 'total_files', 'total_processes', 'ratio_malicious_files', 'ratio_suspicious_processes', 'total_network_activity', 'has_network_threat', 'total_registry', 'risk_score', 'high_risk_flag']

    for col in final_feature_order:
        if col not in df.columns:
            df[col] = 0
    
    return df[final_feature_order]

# ======================================================================================
# 5. DEFINISI HALAMAN-HALAMAN APLIKASI
# ======================================================================================

# Fungsi homepage() tetap sama seperti sebelumnya
def homepage():
    """Menampilkan halaman Homepage yang indah, informatif, dan interaktif."""
    st.title("🛡️ Proyek Analisis & Deteksi Malware Tingkat Lanjut")
    st.markdown("##### Aplikasi Web Cerdas Berbasis *Machine Learning* untuk Klasifikasi Ancaman Siber")
    st.markdown("---")

    with st.container(border=True):
        st.header("📖 Latar Belakang & Tujuan Proyek")
        st.markdown("""
        Di era digital saat ini, ancaman siber seperti *malware, ransomware,* dan *trojan* terus berevolusi, menjadi semakin canggih dan merusak. Kerugian finansial, pencurian data, dan gangguan operasional menjadi risiko nyata bagi individu maupun organisasi. Oleh karena itu, kemampuan untuk mendeteksi dan mengklasifikasikan ancaman ini secara cepat dan akurat adalah pilar utama dalam strategi keamanan siber modern.
        
        **Tujuan Utama Proyek Ini:**
        1.  **Menganalisis Karakteristik Malware**: Melakukan Analisis Data Eksplorasi (EDA) yang mendalam pada dataset file *Portable Executable* (PE) untuk memahami pola dan ciri khas dari file jinak (*Benign*) dan berbagai jenis file berbahaya (*Malicious*).
        2.  **Membangun Model Prediktif Unggul**: Mengembangkan dan melatih model *Machine Learning* (XGBoost) yang memiliki performa tinggi dalam mengklasifikasikan file ke dalam kategori spesifik: **Benign, Ransomware, RAT (Remote Access Trojan), Stealer, dan Trojan**.
        3.  **Menyediakan Alat Interaktif & Edukatif**: Mengimplementasikan model tersebut ke dalam aplikasi web ini, menyediakan antarmuka yang ramah pengguna untuk analisis real-time, serta memberikan wawasan melalui visualisasi data yang interaktif.
        """)

    with st.container(border=True):
        st.header("⚙️ Metodologi & Alur Kerja Proyek")
        # Menggunakan placeholder gambar jika tidak ada file gambar
        try:
            st.image("workflow_diagram.png", caption="Diagram Alur Kerja dari Pengumpulan Data hingga Deployment Aplikasi")
        except:
            st.info("Diagram alur kerja akan ditampilkan di sini.")
        st.markdown("""
        1.  **Pengumpulan Data**: Menggunakan dataset `Datasets_Malware.csv` yang kaya akan fitur.
        2.  **Analisis Data Eksplorasi (EDA)**: Memvisualisasikan distribusi data dan korelasi antar fitur.
        3.  **Preprocessing & Feature Engineering**: Membersihkan data, melakukan *encoding*, dan menciptakan fitur baru seperti `risk_score`.
        4.  **Penyeimbangan Data**: Mengatasi ketidakseimbangan kelas menggunakan teknik **SMOTE**.
        5.  **Pemilihan & Pelatihan Model**: **XGBoost** terpilih sebagai model final berkat performa superiornya.
        6.  **Deployment**: Membungkus model ke dalam aplikasi Streamlit ini.
        """)
    
    st.header("📊 Analisis Eksplorasi Data (EDA) Interaktif")
    df = load_dataset('../Datasets/Datasets_Malware.csv')

    tab1, tab2, tab3 = st.tabs([
        "📝 **Ringkasan Dataset**", 
        "⚖️ **Distribusi Kelas**", 
        "🔬 **Analisis Fitur Penting**"
    ])

    with tab1:
        st.subheader("Gambaran Umum Dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Sampel Total", f"{df.shape[0]:,}")
        col2.metric("Jumlah Fitur Awal", f"{df.shape[1]}")
        col3.metric("Jumlah Kategori Unik", df['Category'].nunique())
        col4.metric("Jumlah Keluarga Malware", df[df['Family'] != 'Benign']['Family'].nunique())
        st.dataframe(df.head(10), hide_index=True)

    with tab2:
        st.subheader("Distribusi Kategori dan Kelas Malware")
        col1, col2 = st.columns([1, 1])
        with col1:
            category_counts = df['Category'].value_counts()
            fig = px.bar(category_counts, x=category_counts.index, y=category_counts.values, title="<b>Distribusi Kategori Malware</b>", labels={'x': 'Kategori', 'y': 'Jumlah Sampel'}, color=category_counts.index, template='plotly_white', text_auto=True)
            fig.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            class_counts = df['Class'].value_counts()
            fig = px.pie(class_counts, names=class_counts.index, values=class_counts.values, title="<b>Distribusi Kelas (Benign vs Malware)</b>", hole=0.4, color_discrete_map={'Benign': '#2ecc71', 'Malware': '#e74c3c'})
            fig.update_traces(textinfo='percent+label', pull=[0, 0.05])
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Top 10 Keluarga Malware Terdeteksi")
        top_families = df[df['Family'] != 'Benign']['Family'].value_counts().head(10)
        fig = px.bar(top_families, y=top_families.index, x=top_families.values, orientation='h', title="<b>Top 10 Keluarga Malware</b>", labels={'y': 'Keluarga Malware', 'x': 'Jumlah Sampel'}, color=top_families.values, color_continuous_scale='plasma', template='plotly_white', text_auto='.2s')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)


def dashboard():
    """Menampilkan halaman Dashboard untuk prediksi interaktif."""
    st.title("🚀 Dasbor Prediksi Kategori Malware")
    st.markdown("Unggah file CSV atau masukkan data tunggal untuk mendapatkan prediksi klasifikasi malware secara instan.")
    
    model = load_model('../Models/xgboost_malware_classifier.joblib')
    category_mapping = {0: 'Benign', 1: 'RAT', 2: 'Ransomware', 3: 'Stealer', 4: 'Trojan'}

    with st.container(border=True):
        st.subheader("Pilih Metode Input Data")
        input_method = st.radio("Metode:", ('Unggah File CSV', 'Input Data Tunggal'), horizontal=True, label_visibility="collapsed")
        
        input_df = None

        if input_method == 'Unggah File CSV':
            uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
            if uploaded_file:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = input_df.copy()
                    st.success(f"File `{uploaded_file.name}` berhasil diunggah ({len(input_df)} baris).")
                except Exception as e:
                    st.error(f"Gagal membaca file CSV: {e}")
        else:
            st.subheader("Input Data Tunggal (format CSV, tanpa header)")
            sample_data = "d5aa2b2506dd71b65307bb551a6a099d,62f4d55a1bb8396e493d7815dc44febed7161535,exe,0x108ec,PE32+,AMD AMD64,MZ,0x0090,0x0003,0x0000,0x0004,0x0000,0xFFFF,0x0000,0x00B8,0x0000,0x0000,0x0000,0x0000,0x000000F8,PE32+,0x00011200,0x0000D200,0x00000000,0x00000000000108EC,0x00001000,0x00004000,0x0000000140000000,0x00001000,0x00000200,5.2,0.0,0x00022000,0x00000400,0x0001E8D0,IMAGE_SUBSYSTEM_WINDOWS_GUI,\"[IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE]\",0x100000,0x1000,0x1000,0x100000,0,0x111FE,0x1000,0x11200,0x400,0,0,\"[IMAGE_SCN_CNT_CODE]\",0x7B14,0x13000,0x7C00,0x11600,0,0,\"[IMAGE_SCN_CNT_INITIALIZED_DATA]\",0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,162.0"
            single_row_input = st.text_area("Tempelkan data satu baris Anda di sini:", value=sample_data, height=100)
            
            if single_row_input:
                original_cols_with_id_no_target = ['md5','sha1','file_extension','EntryPoint','PEType','MachineType','magic_number','bytes_on_last_page','pages_in_file','relocations','size_of_header','min_extra_paragraphs','max_extra_paragraphs','init_ss_value','init_sp_value','init_ip_value','init_cs_value','over_lay_number','oem_identifier','address_of_ne_header','Magic','SizeOfCode','SizeOfInitializedData','SizeOfUninitializedData','AddressOfEntryPoint','BaseOfCode','BaseOfData','ImageBase','SectionAlignment','FileAlignment','OperatingSystemVersion','ImageVersion','SizeOfImage','SizeOfHeaders','Checksum','Subsystem','DllCharacteristics','SizeofStackReserve','SizeofStackCommit','SizeofHeapCommit','SizeofHeapReserve','LoaderFlags','text_VirtualSize','text_VirtualAddress','text_SizeOfRawData','text_PointerToRawData','text_PointerToRelocations','text_PointerToLineNumbers','text_Characteristics','rdata_VirtualSize','rdata_VirtualAddress','rdata_SizeOfRawData','rdata_PointerToRawData','rdata_PointerToRelocations','rdata_PointerToLineNumbers','rdata_Characteristics','registry_read','registry_write','registry_delete','registry_total','network_threats','network_dns','network_http','network_connections','processes_malicious','processes_suspicious','processes_monitored','total_procsses','files_malicious','files_suspicious','files_text','files_unknown','dlls_calls','apis']
                data_list = [val.strip() for val in single_row_input.split(',')]
                
                if len(data_list) == len(original_cols_with_id_no_target):
                    input_df = pd.DataFrame([data_list], columns=original_cols_with_id_no_target)
                    st.session_state.uploaded_data = input_df.copy()
                else:
                    st.error(f"Jumlah kolom tidak sesuai. Diharapkan {len(original_cols_with_id_no_target)}, Anda memberikan {len(data_list)}.")

    if st.button("Analisis & Prediksi Sekarang", type="primary", use_container_width=True):
        if st.session_state.get('uploaded_data') is not None:
            with st.spinner("Model sedang bekerja..."):
                original_data = st.session_state.uploaded_data.copy()
                
                # Preprocessing dengan fungsi yang sudah diperbaiki
                processed_df = preprocess_for_prediction(original_data)
                
                # Prediksi
                predictions_encoded = model.predict(processed_df)
                predictions_proba = model.predict_proba(processed_df)
                
                # Menyiapkan hasil
                results_df = original_data.copy()
                results_df['Prediksi_Kategori'] = [category_mapping.get(p, "Unknown") for p in predictions_encoded]
                results_df['Skor_Kepercayaan'] = [f"{prob.max()*100:.2f}%" for prob in predictions_proba]

                st.session_state.prediction_results = results_df
        else:
            st.warning("⚠️ Harap unggah file atau masukkan data terlebih dahulu.")

    if st.session_state.get('prediction_results') is not None:
        st.markdown("---")
        st.success("Analisis Selesai!")
        st.header("📈 Hasil Prediksi & Analisis")
        
        results_df = st.session_state.prediction_results
        
        st.subheader("Ringkasan Statistik Hasil Prediksi")
        prediction_counts = results_df['Prediksi_Kategori'].value_counts()
        fig = px.pie(prediction_counts, names=prediction_counts.index, values=prediction_counts.values, title="<b>Distribusi Hasil Prediksi</b>", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Detail Hasil Prediksi per Baris")
        display_cols = ['Prediksi_Kategori', 'Skor_Kepercayaan'] + [c for c in results_df.columns if c not in ['Prediksi_Kategori', 'Skor_Kepercayaan']]
        st.dataframe(results_df[display_cols], hide_index=True)


# ======================================================================================
# 6. LOGIKA UTAMA APLIKASI (SIDEBAR & ROUTING)
# ======================================================================================

with st.sidebar:
    # Menghapus logo dan hanya menampilkan judul
    st.title("🛡️ Deteksi Malware")
    
    st.header("Menu Navigasi")

    page = st.selectbox(
        "Pilih Halaman:",
        ["🏠 Homepage", "🚀 Dasbor Prediksi"],
        index=0,  # default ke Homepage
        label_visibility="collapsed"
)
    
    st.markdown("---")
    st.subheader("Tentang Proyek")
    st.info("""
    Aplikasi ini adalah implementasi proyek Keamanan Data untuk mendeteksi
    berbagai jenis malware menggunakan model XGBoost.
    """)

    st.subheader("Anggota Kelompok")
    st.markdown("""
    - **36230022** - Xander Yohanes Dharma
    - **36230031** - Leon Hiunata
    - **36230035** - Josia Given Santoso
    - **36230037** - Vinsensius Erik
    """)

# Logika Routing Halaman
if page == "🏠 Homepage":
    homepage()
elif page == "🚀 Dasbor Prediksi":
    dashboard()
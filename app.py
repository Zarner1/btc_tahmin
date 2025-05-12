# app.py
import streamlit as st
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
import joblib
import os

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="BTC Fiyat Yönü Tahmini", layout="wide")

# --- Configuration ---
MODEL_PATH = "svm_rfe_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "rfe_selected_features.pkl"
# YENİ: Scaler için doğru sütun sırasının yolu
SCALER_COLUMNS_PATH = "features_columns_for_scaler.pkl"
HISTORICAL_DATA_PATH = "C:/Users/yildi/OneDrive/Desktop/projeCrypto/btc_verisi.csv" # Gerekirse güncelleyin
REQUIRED_HISTORICAL_DAYS = 40 # İndikatör hesaplama bağlamı için

# --- 1. Load Saved Artifacts ---
@st.cache_resource
def load_model_assets():
    paths = {
        "Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "RFE Özellikleri": FEATURES_PATH,
        "Scaler Sütunları": SCALER_COLUMNS_PATH
    }
    loaded_assets = {}
    all_exist = True
    for name, path in paths.items():
        if not os.path.exists(path):
            st.error(f"{name} dosyası bulunamadı: {path}")
            all_exist = False
    if not all_exist:
        return None

    try:
        loaded_assets["model"] = joblib.load(MODEL_PATH)
        loaded_assets["scaler"] = joblib.load(SCALER_PATH)
        loaded_assets["rfe_features_list"] = joblib.load(FEATURES_PATH)
        loaded_assets["scaler_columns_list"] = joblib.load(SCALER_COLUMNS_PATH)
        return loaded_assets
    except Exception as e:
        st.error(f"Model varlıkları yüklenirken hata: {e}")
        return None

@st.cache_data # Veri yüklemeyi önbelleğe al
def load_historical_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Geçmiş veri dosyası bulunamadı: {file_path}")
        return None
    try:
        # İndikatör hesaplama bağlamı için sadece Open, Close, Volume gerekli
        df = pd.read_csv(file_path, usecols=['Open', 'Close', 'Volume'])
        if len(df) < REQUIRED_HISTORICAL_DAYS:
            st.warning(f"Geçmiş veri dosyasında yetersiz satır var. En az {REQUIRED_HISTORICAL_DAYS} gün gerekli.")
            # Yine de devam etmeye izin ver, ancak daha sonra hata verebilir
        return df
    except Exception as e:
        st.error(f"Geçmiş veri yüklenirken hata oluştu: {e}")
        return None

# --- 2. Feature Engineering Function ---
def calculate_technical_indicators(df_input):
    df = df_input.copy()
    if 'Close' not in df.columns:
        raise ValueError("DataFrame indikatör hesaplaması için 'Close' sütununu içermelidir.")
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if df['Close'].isnull().any(): # Dönüşüm sonrası kontrol et
        raise ValueError("Close sütunu dönüştürme sonrası sayısal olmayan değerler içeriyor.")

    # RSI
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close=df["Close"])
    df["MACD"] = macd_indicator.macd()
    df["Signal"] = macd_indicator.macd_signal() # MACD Sinyal Hattı

    # SMA20
    sma_indicator = SMAIndicator(close=df["Close"], window=20)
    df["SMA20"] = sma_indicator.sma_indicator()

    # EMA20
    ema_indicator = EMAIndicator(close=df["Close"], window=20)
    df["EMA20"] = ema_indicator.ema_indicator()

    return df

# --- Varlıkları yükle (set_page_config ve fonksiyon tanımlamalarından sonra) ---
assets = load_model_assets()
historical_data_full = load_historical_data(HISTORICAL_DATA_PATH)

# --- 3. Streamlit UI ---
st.title("Bitcoin (BTC/USD) Fiyat Yönü Tahmini")

if assets is None or historical_data_full is None:
    st.error("Uygulama başlatılamadı. Lütfen yukarıdaki hata mesajlarını kontrol edin ve gerekli dosyaların mevcut olduğundan emin olun.")
    st.stop() # Hata varsa uygulamayı durdur

model = assets["model"]
scaler = assets["scaler"]
RFE_SELECTED_FEATURES = assets["rfe_features_list"]
SCALER_INPUT_COLUMNS = assets["scaler_columns_list"] # Dosyadan yüklenen liste

st.sidebar.header("Güncel Günün Verilerini Girin")
st.sidebar.markdown("Lütfen tahmin yapmak istediğiniz günün Açılış (Open), Kapanış (Close) ve Hacim (Volume) değerlerini girin.")

# Gerçekçi varsayılan değerler
default_open = historical_data_full['Open'].iloc[-1] if not historical_data_full.empty else 40000.0
default_close = historical_data_full['Close'].iloc[-1] if not historical_data_full.empty else 40500.0
default_volume = historical_data_full['Volume'].iloc[-1] if not historical_data_full.empty else 50000000000.0

current_open = st.sidebar.number_input("Bugünkü Açılış (Open)", value=float(default_open), step=100.0, format="%.2f")
current_close = st.sidebar.number_input("Bugünkü Kapanış (Close)", value=float(default_close), step=100.0, format="%.2f")
current_volume = st.sidebar.number_input("Bugünkü Hacim (Volume)", value=float(default_volume), step=1000000.0, format="%.0f")

if st.sidebar.button("Tahmin Et", type="primary"):
    if len(historical_data_full) < REQUIRED_HISTORICAL_DAYS -1 :
         st.error(f"Tahmin için yetersiz geçmiş veri. En az {REQUIRED_HISTORICAL_DAYS-1} geçmiş gün verisi gereklidir.")
    else:
        try:
            # Bağlam için geçmiş verinin son N-1 gününü al
            recent_history = historical_data_full[['Open', 'Close', 'Volume']].iloc[-(REQUIRED_HISTORICAL_DAYS - 1):].copy()

            # Mevcut günün verileri için bir DataFrame oluştur
            current_data_dict = {
                'Open': [current_open],
                'Close': [current_close],
                'Volume': [current_volume]
            }
            current_df_row = pd.DataFrame(current_data_dict)

            # Geçmiş verileri mevcut günle birleştir
            combined_df = pd.concat([recent_history, current_df_row], ignore_index=True)

            # İndikatörleri hesapla
            df_with_indicators = calculate_technical_indicators(combined_df)

            # Son satırı al (tüm indikatörlerle mevcut günün verileri)
            current_features_for_prediction = df_with_indicators.iloc[-1:].copy() # DataFrame olarak kalsın

            # SCALER_INPUT_COLUMNS artık dosyadan yükleniyor.
            
            # Ölçekleme için tüm gerekli sütunların mevcut olduğundan emin ol
            missing_cols = [col for col in SCALER_INPUT_COLUMNS if col not in current_features_for_prediction.columns]
            if missing_cols:
                st.error(f"İndikatör hesaplaması sonrası beklenen sütunlar eksik: {', '.join(missing_cols)}. "
                         f"Beklenen sütunlar: {SCALER_INPUT_COLUMNS}. "
                         f"Mevcut sütunlar: {current_features_for_prediction.columns.tolist()}")
            # current_features_for_prediction'dan SCALER_INPUT_COLUMNS kullanarak bir alt küme oluşturmadan önce NaN kontrolü yap
            elif current_features_for_prediction.isnull().values.any(): # İndikatör hesaplamasından sonra NaN var mı kontrol et
                 st.error("İndikatör hesaplaması sonucu NaN değerler oluştu. Bu genellikle yetersiz geçmiş veriden kaynaklanır. Lütfen verileri kontrol edin.")
                 st.write("İndikatörlü son satır (NaN içeren):")
                 st.dataframe(current_features_for_prediction)

            elif current_features_for_prediction[SCALER_INPUT_COLUMNS].isnull().values.any(): # Daha sağlam NaN kontrolü
                st.error("Teknik indikatörler hesaplanırken bir sorun oluştu (NaN değerler). Bu genellikle yetersiz geçmiş veriden veya girilen hatalı değerlerden kaynaklanır.")
                st.write("Hesaplanan son satır (indikatörler dahil, ölçekleme öncesi):")
                st.dataframe(current_features_for_prediction[SCALER_INPUT_COLUMNS])
            else:
                # Özellikleri scaler'ın beklediği sıraya göre düzenle
                to_scale_df = current_features_for_prediction[SCALER_INPUT_COLUMNS]
                
                # Özellikleri ölçekle
                scaled_features_array = scaler.transform(to_scale_df)
                # Ölçeklenmiş DataFrame'i de aynı sütun isimleri ve sırasıyla oluştur
                scaled_features_df = pd.DataFrame(scaled_features_array, columns=SCALER_INPUT_COLUMNS, index=to_scale_df.index)
                
                # RFE ile seçilen özellikleri bu doğru sıralanmış ve ölçeklenmiş DataFrame'den al
                final_features_for_model = scaled_features_df[RFE_SELECTED_FEATURES]

                # Tahmin yap
                prediction = model.predict(final_features_for_model)
                
                # Sonucu göster
                st.subheader("Tahmin Sonucu")
                col1, col2 = st.columns([1,3])
                with col1:
                    if prediction[0] == 1:
                        st.image("https://emojigraph.org/media/apple/chart-increasing_1f4c8.png", width=100)
                    else:
                        st.image("https://emojigraph.org/media/apple/chart-decreasing_1f4c9.png", width=100)
                with col2:
                    if prediction[0] == 1:
                        st.success("📈 **YÜKSELİŞ** bekleniyor.")
                        st.markdown("Tahminimiz, yarınki Bitcoin kapanış fiyatının bugünkü kapanış fiyatından **daha yüksek** olacağı yönündedir.")
                    else:
                        st.error("📉 **DÜŞÜŞ** bekleniyor.")
                        st.markdown("Tahminimiz, yarınki Bitcoin kapanış fiyatının bugünkü kapanış fiyatından **daha düşük** olacağı yönündedir.")

                with st.expander("Detaylar: Modele Giren Veriler"):
                    st.markdown("##### Ham Girilen Veriler (Bugün):")
                    st.dataframe(current_df_row.style.format({"Open": "{:.2f}", "Close": "{:.2f}", "Volume": "{:.0f}"}))
                    st.markdown(f"##### {REQUIRED_HISTORICAL_DAYS} Günlük Veri Üzerinden Hesaplanan İndikatörler (Bugün İçin, Ölçekleme Öncesi):")
                    st.dataframe(to_scale_df.style.format("{:.2f}"))
                    st.markdown("##### Ölçeklenmiş ve Model İçin Seçilmiş Özellikler:")
                    st.dataframe(final_features_for_model.style.format("{:.4f}"))
                    st.caption(f"Modelin kullandığı özellikler: ` {', '.join(RFE_SELECTED_FEATURES)} `")

        except ValueError as ve:
            st.error(f"Veri hatası: {ve}")
        except Exception as e:
            st.error(f"Tahmin sırasında beklenmedik bir hata oluştu: {e}")
            st.error("Lütfen girdiğiniz değerleri ve geçmiş veri dosyasının bütünlüğünü kontrol edin.")
            import traceback
            st.text(traceback.format_exc()) # Hatanın tam dökümünü görmek için

st.sidebar.markdown("---")
st.sidebar.caption("Bu uygulama demo amaçlıdır ve yatırım tavsiyesi değildir.")

# --- İsteğe bağlı: Bazı geçmiş veri bağlamını göster ---
if not historical_data_full.empty:
    st.markdown("---")
    st.subheader("Geçmiş Veri Örneği (Son 5 Gün)")
    st.dataframe(historical_data_full[['Open', 'Close', 'Volume']].tail().reset_index(drop=True).style.format({"Open": "{:.2f}", "Close": "{:.2f}", "Volume": "{:.0f}"}))
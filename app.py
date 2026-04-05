# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:51:02 2026

@author: suueda
"""
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import io
import time
import google.generativeai as genai

st.set_page_config(page_title="Industrial Incident AI Dashboard", layout="wide")

st.title("Industrial Incident AI Decision Support Dashboard (Gemini Powered)")
st.markdown("Excel yükle, Google Gemini AI ile analiz et, sonucu görselleştir.")

st.sidebar.header("Ayarlar")

max_rows = st.sidebar.slider("İşlenecek maksimum satır", min_value=1, max_value=50, value=10)
use_sample = st.sidebar.checkbox("Rastgele örneklem al", value=False)
sleep_between_calls = st.sidebar.slider("İstekler arası bekleme (sn)", 0.5, 5.0, 1.0, 0.5)

# Secret'tan API key çek
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
else:
    model = None

def process_row_gemini(row):
    prompt = f"""
    Sen bir endüstriyel arıza ve bakım uzmanısın. Aşağıdaki arıza kaydını analiz et.
    Cevabı SADECE VE SADECE aşağıda belirtilen JSON formatında ver. Başka açıklama ekleme.

    Arıza Kaydı:
    - Incident ID: {row['incident_id']}
    - Metin: {row['incident_text']}
    - Hat: {row['line']}

    Beklenen JSON Formatı:
    {{
        "fault_category": "Mekanik veya Elektrik veya Yazılım",
        "maintenance_priority": "Low veya Medium veya High veya Critical",
        "human_review_required": "Yes veya No",
        "maintenance_action_plan": "Kısa teknik çözüm önerisi",
        "manager_summary": "Yönetici için tek cümlelik özet",
        "workflow_status": "succeeded"
    }}
    """

    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_text)
        return result
    except Exception as e:
        return {
            "workflow_status": "error",
            "error": str(e),
            "fault_category": "Error",
            "maintenance_priority": "Unknown",
            "human_review_required": "Unknown"
        }

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Processed_Output", index=False)
    return output.getvalue()

uploaded_file = st.file_uploader("Excel dosyasını yükle", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("Yüklenen Veri Önizleme")
    st.dataframe(df.head(5), use_container_width=True)

    required_cols = ["incident_id", "incident_text", "line"]
    if all(c in df.columns for c in required_cols):

        if use_sample:
            df_proc = df.sample(n=min(max_rows, len(df)), random_state=42).copy()
        else:
            df_proc = df.head(max_rows).copy()

        if st.button("Gemini Analizini Başlat", type="primary"):
            if not gemini_api_key:
                st.error("Streamlit Secrets içine GEMINI_API_KEY eklenmemiş.")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (_, row) in enumerate(df_proc.iterrows(), start=1):
                    status_text.write(f"İşleniyor: {i}/{len(df_proc)} | ID: {row['incident_id']}")
                    analysis = process_row_gemini(row)
                    combined_data = {**row.to_dict(), **analysis}
                    results.append(combined_data)
                    progress_bar.progress(i / len(df_proc))
                    time.sleep(sleep_between_calls)

                df_out = pd.DataFrame(results)
                st.success("Analiz başarıyla tamamlandı!")

                st.subheader("Özet Göstergeler")
                c1, c2, c3 = st.columns(3)
                c1.metric("Toplam Kayıt", len(df_out))
                if "maintenance_priority" in df_out.columns:
                    c2.metric("Kritik Arızalar", (df_out["maintenance_priority"] == "Critical").sum())
                    c3.metric("İnceleme Gereken", (df_out["human_review_required"] == "Yes").sum())

                st.dataframe(df_out, use_container_width=True)

                st.download_button(
                    label="Sonuçları Excel Olarak İndir",
                    data=to_excel_bytes(df_out),
                    file_name="gemini_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.error(f"Excel'de gerekli sütunlar bulunamadı! Beklenen: {required_cols}")

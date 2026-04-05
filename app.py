# -*- coding: utf-8 -*-
import io
import json
import time

import pandas as pd
import streamlit as st
from google import genai

st.set_page_config(page_title="Industrial Incident AI Dashboard", layout="wide")

st.title("Industrial Incident AI Decision Support Dashboard (Gemini Powered)")
st.markdown("Excel yükle, Gemini ile analiz et, sonucu görselleştir.")

st.sidebar.header("Ayarlar")
max_rows = st.sidebar.slider("İşlenecek maksimum satır", min_value=1, max_value=50, value=10)
use_sample = st.sidebar.checkbox("Rastgele örneklem al", value=False)
sleep_between_calls = st.sidebar.slider("İstekler arası bekleme (sn)", 0.5, 5.0, 1.0, 0.5)

# Secrets'tan API key çek
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)

# Güncel ve daha güvenli model seçimi
MODEL_NAME = "gemini-2.5-flash"

client = None
if gemini_api_key:
    client = genai.Client(api_key=gemini_api_key)

def clean_json_text(text: str) -> str:
    """Model cevabındaki markdown json fence'lerini temizler."""
    return text.replace("```json", "").replace("```", "").strip()

def process_row_gemini(row: pd.Series) -> dict:
    prompt = f"""
Sen bir endüstriyel arıza ve bakım uzmanısın.

Görevin:
Aşağıdaki olay kaydını analiz edip SADECE geçerli JSON döndürmek.
Başka hiçbir açıklama yazma.
Kategori seçimlerinde mühendislik mantığı kullan.

Olay Kaydı:
- Incident ID: {row['incident_id']}
- Metin: {row['incident_text']}
- Hat: {row['line']}

Kurallar:
- fault_category yalnızca şu değerlerden biri olsun:
  ["mekanik", "elektrik", "operatör", "yağlama", "proses"]
- maintenance_priority yalnızca şu değerlerden biri olsun:
  ["Low", "Medium", "High", "Critical"]
- human_review_required yalnızca şu değerlerden biri olsun:
  ["Yes", "No"]
- Kısa, net ve teknik yaz.
- Eksik bilgi varsa en makul mühendislik çıkarımını yap ama uydurma detay ekleme.

Döndürülecek JSON formatı:
{{
  "fault_category": "mekanik",
  "maintenance_priority": "High",
  "human_review_required": "Yes",
  "maintenance_action_plan": "Kısa teknik aksiyon planı",
  "manager_summary": "Yönetici için tek cümlelik özet",
  "workflow_status": "succeeded"
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        text = clean_json_text(response.text)
        result = json.loads(text)

        # Beklenen alanlar yoksa güvenli fallback
        return {
            "fault_category": result.get("fault_category", "Error"),
            "maintenance_priority": result.get("maintenance_priority", "Unknown"),
            "human_review_required": result.get("human_review_required", "Unknown"),
            "maintenance_action_plan": result.get("maintenance_action_plan", ""),
            "manager_summary": result.get("manager_summary", ""),
            "workflow_status": result.get("workflow_status", "succeeded"),
            "error": ""
        }

    except Exception as e:
        return {
            "workflow_status": "error",
            "error": str(e),
            "fault_category": "Error",
            "maintenance_priority": "Unknown",
            "human_review_required": "Unknown",
            "maintenance_action_plan": "",
            "manager_summary": ""
        }

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Processed_Output", index=False)
    return output.getvalue()

uploaded_file = st.file_uploader("Excel dosyasını yükle", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str).str.strip()
    except Exception as e:
        st.error(f"Excel okunamadı: {e}")
        st.stop()

    st.subheader("Yüklenen Veri Önizleme")
    st.dataframe(df.head(5), use_container_width=True)

    required_cols = ["incident_id", "incident_text", "line"]

    if all(col in df.columns for col in required_cols):
        if use_sample:
            df_proc = df.sample(n=min(max_rows, len(df)), random_state=42).copy()
        else:
            df_proc = df.head(max_rows).copy()

        if st.button("Gemini Analizini Başlat", type="primary"):
            if not gemini_api_key:
                st.error("Streamlit Secrets içine GEMINI_API_KEY eklenmemiş.")
            else:
                st.info(f"Kullanılan model: {MODEL_NAME}")
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

                st.success("Analiz tamamlandı.")

                # Özet göstergeler
                st.subheader("Özet Göstergeler")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam Kayıt", len(df_out))
                c2.metric("Kritik Arıza", (df_out["maintenance_priority"] == "Critical").sum())
                c3.metric("İnsan İnceleme", (df_out["human_review_required"] == "Yes").sum())

                top_fault = "Yok"
                valid_faults = df_out[df_out["fault_category"] != "Error"]["fault_category"]
                if not valid_faults.empty:
                    top_fault = valid_faults.mode().iloc[0]
                c4.metric("En Sık Kategori", top_fault)

                # Grafikler
                st.subheader("Arıza Dağılımı")
                if "fault_category" in df_out.columns:
                    st.bar_chart(df_out["fault_category"].value_counts())

                st.subheader("Bakım Önceliği Dağılımı")
                if "maintenance_priority" in df_out.columns:
                    st.bar_chart(df_out["maintenance_priority"].value_counts())

                # Yönetici özeti
                st.subheader("Yönetici Özeti")
                summary_lines = [f"Toplam {len(df_out)} kayıt analiz edildi."]

                if top_fault != "Yok":
                    summary_lines.append(f"En sık görülen arıza kategorisi: {top_fault}.")

                critical_count = (df_out["maintenance_priority"] == "Critical").sum()
                summary_lines.append(f"Kritik öncelikli kayıt sayısı: {critical_count}.")

                review_count = (df_out["human_review_required"] == "Yes").sum()
                summary_lines.append(f"İnsan incelemesi gereken kayıt sayısı: {review_count}.")

                for line in summary_lines:
                    st.write(f"- {line}")

                st.subheader("Analiz Sonuç Tablosu")
                st.dataframe(df_out, use_container_width=True)

                st.download_button(
                    label="Sonuçları Excel Olarak İndir",
                    data=to_excel_bytes(df_out),
                    file_name="gemini_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.error(f"Excel'de gerekli sütunlar bulunamadı. Beklenen sütunlar: {required_cols}")
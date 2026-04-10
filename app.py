# -*- coding: utf-8 -*-
import io
import json
import time

import pandas as pd
import streamlit as st
from google import genai

st.set_page_config(page_title="LLM Tabanlı Arıza Karar Destek Sistemi", layout="wide")

st.title("LLM Tabanlı Arıza Karar Destek Sistemi")
st.markdown("Excel yükle, LLM tabanlı analiz yap, risk seviyesini hesapla ve sonuçları incele.")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Ayarlar")
max_rows = st.sidebar.slider("İşlenecek maksimum satır", min_value=1, max_value=50, value=10)
use_sample = st.sidebar.checkbox("Rastgele örneklem al", value=False)
sleep_between_calls = st.sidebar.slider("İstekler arası bekleme (sn)", 0.0, 5.0, 0.5, 0.5)
demo_mode = st.sidebar.checkbox("Demo/Fallback Mode", value=True)

gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
MODEL_NAME = "gemini-2.5-flash"

client = None
if gemini_api_key:
    client = genai.Client(api_key=gemini_api_key)

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def clean_json_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("```json", "").replace("```", "").strip()

def safe_json_loads(text: str) -> dict:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        return {}

def normalize_yes_no(val) -> str:
    if pd.isna(val):
        return "No"
    text = str(val).strip().lower()
    if text in ["yes", "y", "evet", "true", "1"]:
        return "Yes"
    return "No"

def normalize_category(val: str) -> str:
    valid = ["Mekanik", "Elektrik", "Operatör", "Yağlama", "Proses"]
    if val in valid:
        return val
    return "Error"

def normalize_severity(val: str) -> str:
    valid = ["Low", "Medium", "High", "Critical"]
    if val in valid:
        return val
    return "Unknown"

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if "line" not in df.columns and "line_no" in df.columns:
        df["line"] = df["line_no"]

    if "incident_text" in df.columns:
        df["incident_text"] = df["incident_text"].astype(str).str.strip()

    if "shift" in df.columns:
        df["shift"] = df["shift"].astype(str).str.strip()

    if "line" in df.columns:
        df["line"] = df["line"].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    optional_defaults = {
        "Downtime_Min": 0,
        "Repeated_Fault": "No",
        "Urgency_Flag": "No",
        "Maintenance_Required": "No",
        "Status": "Open"
    }

    for col, default_val in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["Downtime_Min"] = pd.to_numeric(df["Downtime_Min"], errors="coerce").fillna(0)
    df["Repeated_Fault"] = df["Repeated_Fault"].apply(normalize_yes_no)
    df["Urgency_Flag"] = df["Urgency_Flag"].apply(normalize_yes_no)
    df["Maintenance_Required"] = df["Maintenance_Required"].apply(normalize_yes_no)

    if "incident_text" in df.columns:
        df = df[df["incident_text"].astype(str).str.strip() != ""].copy()

    return df

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham Excel'de bu sütunlar yoksa otomatik ekler.
    Böylece Streamlit'e yüklediğin dosya, analiz sonrası genişletilmiş hale gelir.
    """
    df = df.copy()

    output_defaults = {
        "standard_fault_category": "",
        "technical_summary": "",
        "probable_root_cause": "",
        "human_intervention_needed": "",
        "severity_comment": "",
        "risk_score": 0,
        "risk_level": "",
        "maintenance_priority": "",
        "maintenance_action_plan": "",
        "manager_summary": "",
        "llm1_status": "",
        "llm1_error": "",
        "llm2_status": "",
        "llm2_error": ""
    }

    for col, default_val in output_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    return df

# =========================
# FALLBACK ANALİZ
# =========================
def local_fallback_llm1(row: pd.Series) -> dict:
    text = str(row.get("incident_text", "")).lower()

    if any(k in text for k in ["kıvılcım", "panel", "elektrik", "yanık"]):
        category = "Elektrik"
        root = "Elektrik bağlantılarında gevşeme, kısa devre veya panel kaynaklı sorun olabilir."
        severity = "Critical" if any(k in text for k in ["kıvılcım", "yanık"]) else "High"
        human = "Yes"
    elif any(k in text for k in ["yağ", "yağlama", "kuru çalışma"]):
        category = "Yağlama"
        root = "Yetersiz yağlama veya yağlama planı eksikliği olabilir."
        severity = "High"
        human = "Yes"
    elif any(k in text for k in ["operatör", "yanlış", "kapak açık"]):
        category = "Operatör"
        root = "Operasyon prosedürü hatası veya kullanıcı kaynaklı işlem hatası olabilir."
        severity = "Medium"
        human = "Yes"
    elif any(k in text for k in ["akış", "çevrim", "proses", "birikme"]):
        category = "Proses"
        root = "Proses akışında dengesizlik veya darboğaz oluşmuş olabilir."
        severity = "Medium"
        human = "No"
    else:
        category = "Mekanik"
        root = "Mekanik aşınma, gevşeme veya hizasızlık olabilir."
        severity = "High" if any(k in text for k in ["titreşim", "ısındı", "durdu"]) else "Medium"
        human = "Yes" if severity in ["High", "Critical"] else "No"

    return {
        "standard_fault_category": category,
        "technical_summary": f"Olay kaydı {category.lower()} kaynaklı teknik bir arızaya işaret ediyor.",
        "probable_root_cause": root,
        "human_intervention_needed": human,
        "severity_comment": severity,
        "llm1_status": "fallback",
        "llm1_error": ""
    }

def local_fallback_llm2(row: pd.Series, llm1_output: dict, rule_output: dict) -> dict:
    category = llm1_output.get("standard_fault_category", "Arıza")
    risk = rule_output.get("risk_level", "Medium")
    action = "İlgili ekipmanı kontrol et, güvenli müdahale prosedürünü uygula ve bakım kaydı oluştur."
    summary = f"{category} kategorisinde bir olay tespit edildi. Risk seviyesi {risk} olarak değerlendirildi."
    return {
        "maintenance_action_plan": action,
        "manager_summary": summary,
        "llm2_status": "fallback",
        "llm2_error": ""
    }

# =========================
# LLM1 — ANALİZ
# =========================
def analyze_with_llm_1(row: pd.Series) -> dict:
    prompt = f"""
Sen endüstriyel bakım ve operasyon verilerini analiz eden teknik bir asistansın.

Görevin:
Aşağıdaki arıza kaydını analiz et ve SADECE geçerli JSON döndür.
Paragraf, açıklama, markdown veya ek metin yazma.

Olay kaydı:
- incident_id: {row.get('incident_id', '')}
- date: {row.get('date', '')}
- shift: {row.get('shift', '')}
- line: {row.get('line', '')}
- incident_text: {row.get('incident_text', '')}

Kurallar:
- standard_fault_category aşağıdakilerden biri olsun:
  ["Mekanik", "Elektrik", "Operatör", "Yağlama", "Proses"]
- human_intervention_needed yalnızca:
  ["Yes", "No"]
- severity_comment aşağıdakilerden biri olsun:
  ["Low", "Medium", "High", "Critical"]
- Teknik özet kısa ve net olsun.
- Kök neden tahmini makul olsun ama gereksiz detay uydurma.

JSON formatı:
{{
  "standard_fault_category": "Mekanik",
  "technical_summary": "Kısa teknik özet",
  "probable_root_cause": "Muhtemel kök neden",
  "human_intervention_needed": "Yes",
  "severity_comment": "High"
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        result = safe_json_loads(getattr(response, "text", ""))

        if not result:
            raise ValueError("LLM1 geçerli JSON döndürmedi.")

        return {
            "standard_fault_category": normalize_category(result.get("standard_fault_category", "Error")),
            "technical_summary": result.get("technical_summary", ""),
            "probable_root_cause": result.get("probable_root_cause", ""),
            "human_intervention_needed": normalize_yes_no(result.get("human_intervention_needed", "No")),
            "severity_comment": normalize_severity(result.get("severity_comment", "Unknown")),
            "llm1_status": "success",
            "llm1_error": ""
        }

    except Exception as e:
        if demo_mode:
            fallback = local_fallback_llm1(row)
            fallback["llm1_error"] = str(e)
            return fallback

        return {
            "standard_fault_category": "Error",
            "technical_summary": "",
            "probable_root_cause": "",
            "human_intervention_needed": "No",
            "severity_comment": "Unknown",
            "llm1_status": "error",
            "llm1_error": str(e)
        }

# =========================
# RULE ENGINE
# =========================
def apply_rules(row: pd.Series, llm1_output: dict) -> dict:
    score = 0

    if row.get("Downtime_Min", 0) > 60:
        score += 3

    if normalize_yes_no(row.get("Repeated_Fault", "No")) == "Yes":
        score += 2

    if normalize_yes_no(row.get("Urgency_Flag", "No")) == "Yes":
        score += 2

    if llm1_output.get("human_intervention_needed", "No") == "Yes":
        score += 2

    severity_comment = str(llm1_output.get("severity_comment", "")).lower()
    if severity_comment in ["critical", "high"]:
        score += 2

    incident_text = str(row.get("incident_text", "")).lower()
    if any(x in incident_text for x in ["durdu", "alarm", "kıvılcım", "ısındı", "yanık", "duman"]):
        score += 1

    if score <= 2:
        level = "Low"
    elif score <= 5:
        level = "Medium"
    else:
        level = "High"

    if level == "High":
        priority = "Critical"
    elif level == "Medium":
        priority = "High"
    else:
        priority = "Medium"

    return {
        "risk_score": score,
        "risk_level": level,
        "maintenance_priority": priority
    }

# =========================
# LLM2 — AKSİYON
# =========================
def generate_action_with_llm_2(row: pd.Series, llm1_output: dict, rule_output: dict) -> dict:
    prompt = f"""
Sen endüstriyel bakım planlama ve karar destek alanında uzman teknik bir asistansın.

Görevin:
Aşağıdaki analiz sonuçlarına göre SADECE geçerli JSON döndür.
Paragraf, markdown veya ek açıklama yazma.

Veriler:
- Kategori: {llm1_output.get("standard_fault_category", "")}
- Teknik özet: {llm1_output.get("technical_summary", "")}
- Muhtemel kök neden: {llm1_output.get("probable_root_cause", "")}
- Human intervention needed: {llm1_output.get("human_intervention_needed", "")}
- Severity: {llm1_output.get("severity_comment", "")}
- Risk level: {rule_output.get("risk_level", "")}
- Risk score: {rule_output.get("risk_score", "")}
- Maintenance priority: {rule_output.get("maintenance_priority", "")}

Kurallar:
- maintenance_action_plan kısa, teknik ve uygulanabilir olsun.
- manager_summary yönetici için tek cümlelik net özet olsun.

JSON formatı:
{{
  "maintenance_action_plan": "Önerilen teknik aksiyon",
  "manager_summary": "Yönetici özeti"
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        result = safe_json_loads(getattr(response, "text", ""))

        if not result:
            raise ValueError("LLM2 geçerli JSON döndürmedi.")

        return {
            "maintenance_action_plan": result.get("maintenance_action_plan", ""),
            "manager_summary": result.get("manager_summary", ""),
            "llm2_status": "success",
            "llm2_error": ""
        }

    except Exception as e:
        if demo_mode:
            fallback = local_fallback_llm2(row, llm1_output, rule_output)
            fallback["llm2_error"] = str(e)
            return fallback

        return {
            "maintenance_action_plan": "",
            "manager_summary": "",
            "llm2_status": "error",
            "llm2_error": str(e)
        }

def build_manager_summary(df: pd.DataFrame) -> list[str]:
    lines = []
    total_records = len(df)
    high_risk_count = int((df["risk_level"] == "High").sum())
    top_fault = "Yok"

    valid_faults = df[df["standard_fault_category"] != "Error"]["standard_fault_category"]
    if not valid_faults.empty:
        top_fault = valid_faults.mode().iloc[0]

    lines.append(f"Toplam {total_records} kayıt analiz edildi.")
    lines.append(f"High-risk kayıt sayısı: {high_risk_count}.")
    lines.append(f"En sık görülen arıza kategorisi: {top_fault}.")

    top_lines = df["line"].value_counts().head(3)
    if not top_lines.empty:
        joined = ", ".join([f"{idx} ({val})" for idx, val in top_lines.items()])
        lines.append(f"Yoğunlaşma görülen hatlar: {joined}.")

    return lines

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Ham + zenginleşmiş sonuç aynı sheet'te
        df.to_excel(writer, sheet_name="Analyzed_Results", index=False)
    return output.getvalue()

# =========================
# UI
# =========================
uploaded_file = st.file_uploader("Excel dosyasını yükle", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df = preprocess_dataframe(df)
        df = ensure_output_columns(df)  # <-- yüklenen Excel'e sistem kolonlarını ekliyor
    except Exception as e:
        st.error(f"Excel okunamadı: {e}")
        st.stop()

    required_cols = ["incident_id", "date", "shift", "line", "incident_text"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.error(f"Eksik zorunlu kolonlar: {missing_cols}")
        st.stop()

    st.subheader("Ekran 1 — Veri Önizleme")
    st.dataframe(df.head(10), width="stretch")

    if use_sample:
        df_proc = df.sample(n=min(max_rows, len(df)), random_state=42).copy()
    else:
        df_proc = df.head(max_rows).copy()

    st.info(f"İşlenecek satır sayısı: {len(df_proc)}")

    if st.button("LLM Analizini Başlat", type="primary"):
        if not gemini_api_key and not demo_mode:
            st.error("GEMINI_API_KEY yok. Demo mode kapalı olduğu için analiz başlatılamıyor.")
            st.stop()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (_, row) in enumerate(df_proc.iterrows(), start=1):
            status_text.write(f"İşleniyor: {i}/{len(df_proc)} | ID: {row.get('incident_id', '')}")

            llm1_output = analyze_with_llm_1(row)
            rule_output = apply_rules(row, llm1_output)
            llm2_output = generate_action_with_llm_2(row, llm1_output, rule_output)

            combined = {
                **row.to_dict(),
                **llm1_output,
                **rule_output,
                **llm2_output
            }

            results.append(combined)
            progress_bar.progress(i / len(df_proc))
            time.sleep(sleep_between_calls)

        df_final = pd.DataFrame(results)

        # =========================
        # EKRAN 2 — GENEL ÖZET
        # =========================
        st.subheader("Ekran 2 — Genel Özet")

        total_records = len(df_final)
        total_downtime = float(df_final["Downtime_Min"].sum())
        high_risk_count = int((df_final["risk_level"] == "High").sum())

        top_fault = "Yok"
        valid_faults = df_final[df_final["standard_fault_category"] != "Error"]["standard_fault_category"]
        if not valid_faults.empty:
            top_fault = valid_faults.mode().iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Kayıt", total_records)
        c2.metric("Toplam Duruş Süresi", int(total_downtime))
        c3.metric("High-Risk Kayıt", high_risk_count)
        c4.metric("En Sık Arıza Kategorisi", top_fault)

        # =========================
        # EKRAN 3 — FİLTRELENEBİLİR TABLO
        # =========================
        st.subheader("Ekran 3 — Filtrelenebilir Tablo")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            selected_shift = st.selectbox("Vardiya", ["Tümü"] + sorted(df_final["shift"].astype(str).unique().tolist()))
        with filter_col2:
            selected_line = st.selectbox("Hat", ["Tümü"] + sorted(df_final["line"].astype(str).unique().tolist()))
        with filter_col3:
            selected_risk = st.selectbox("Risk Seviyesi", ["Tümü"] + sorted(df_final["risk_level"].astype(str).unique().tolist()))
        with filter_col4:
            selected_human = st.selectbox("İnsan Müdahalesi", ["Tümü"] + sorted(df_final["human_intervention_needed"].astype(str).unique().tolist()))

        filtered_df = df_final.copy()

        if selected_shift != "Tümü":
            filtered_df = filtered_df[filtered_df["shift"].astype(str) == selected_shift]

        if selected_line != "Tümü":
            filtered_df = filtered_df[filtered_df["line"].astype(str) == selected_line]

        if selected_risk != "Tümü":
            filtered_df = filtered_df[filtered_df["risk_level"].astype(str) == selected_risk]

        if selected_human != "Tümü":
            filtered_df = filtered_df[filtered_df["human_intervention_needed"].astype(str) == selected_human]

        st.dataframe(filtered_df, width="stretch")

        # =========================
        # EKRAN 4 — KAYIT DETAY ANALİZİ
        # =========================
        st.subheader("Ekran 4 — Kayıt Detay Analizi")

        record_options = filtered_df["incident_id"].astype(str).tolist()
        if record_options:
            selected_record = st.selectbox("Kayıt Seç", record_options)
            detail_row = filtered_df[filtered_df["incident_id"].astype(str) == selected_record].iloc[0]

            d1, d2 = st.columns(2)

            with d1:
                st.markdown("**Ham Not**")
                st.write(detail_row["incident_text"])

                st.markdown("**LLM1 Teknik Özeti**")
                st.write(detail_row["technical_summary"])

                st.markdown("**Muhtemel Kök Neden**")
                st.write(detail_row["probable_root_cause"])

                st.markdown("**Arıza Kategorisi**")
                st.write(detail_row["standard_fault_category"])

            with d2:
                st.markdown("**Risk Seviyesi**")
                st.write(detail_row["risk_level"])

                st.markdown("**Risk Skoru**")
                st.write(detail_row["risk_score"])

                st.markdown("**Bakım Önceliği**")
                st.write(detail_row["maintenance_priority"])

                st.markdown("**İnsan Müdahalesi Gerekli mi?**")
                st.write(detail_row["human_intervention_needed"])

                st.markdown("**LLM2 Aksiyon Planı**")
                st.write(detail_row["maintenance_action_plan"])

                st.markdown("**Yönetici Özeti**")
                st.write(detail_row["manager_summary"])

        # =========================
        # EKRAN 5 — YÖNETİCİ ÖZETİ
        # =========================
        st.subheader("Ekran 5 — Yönetici Özeti")

        for line in build_manager_summary(df_final):
            st.write(f"- {line}")

        high_risk_df = df_final[df_final["risk_level"] == "High"].copy()
        top_5_critical = high_risk_df.head(5)

        st.markdown("**En Kritik 5 Olay**")
        if len(top_5_critical) > 0:
            st.dataframe(
                top_5_critical[
                    [
                        "incident_id",
                        "line",
                        "standard_fault_category",
                        "risk_score",
                        "risk_level",
                        "maintenance_action_plan"
                    ]
                ],
                width="stretch"
            )
        else:
            st.info("High-risk kayıt bulunmadı.")

        st.markdown("**Tekrar Eden Patternler**")
        st.bar_chart(df_final["standard_fault_category"].value_counts())

        st.markdown("**Hat Bazlı Yoğunlaşma**")
        st.bar_chart(df_final["line"].value_counts())

        st.markdown("**Kısa Aksiyon Listesi**")
        action_df = (
            df_final[["incident_id", "maintenance_action_plan", "risk_level"]]
            .drop_duplicates()
            .head(10)
        )
        st.dataframe(action_df, width="stretch")

        st.download_button(
            label="Sonuçları Excel Olarak İndir",
            data=to_excel_bytes(df_final),
            file_name="llm_two_stage_analyzed_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Başlamak için bir Excel dosyası yükle.")
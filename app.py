import streamlit as st
import numpy as np
import pickle
import time

# ====================================================================
# 1. إعدادات الصفحة العامة وتنسيقات CSS المفتوحة والنظيفة
# ====================================================================
st.set_page_config(
    page_title="AI Smart Irrigation Assessment System",
    page_icon="🌱",
    layout="centered"
)

# كود CSS مخصص لتنظيم العناصر وجعل الواجهة مفتوحة تماماً مع إجبار الأرقام الإنجليزية
st.markdown("""
    <style>
    /* إخفاء القوائم الجانبية الافتراضية للحفاظ على جمالية الموقع كصفحة مستقلة */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    .stApp { background-color: #F8F9FA; font-family: sans-serif; }
    
    /* تنسيق حاوية الصفحة الأولى الرئيسية */
    .welcome-container { text-align: center; padding: 40px 20px; margin-top: 8vh; }
    .main-title { font-size: 38px; font-weight: 800; color: #1B5E20; margin-bottom: 5px; }
    .sub-title { font-size: 14px; font-weight: bold; color: #81C784; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 40px; }
    
    /* تنسيق السطر المستقيم للأسماء في الصفحة الأولى */
    .team-line-main { font-size: 16px; font-weight: 700; color: #2E7D32; text-align: center; margin-bottom: 8px; }
    .supervisor-line-main { font-size: 15px; font-weight: 600; color: #555555; text-align: center; margin-bottom: 35px; }
    
    /* التنسيق المفتوح لخانات الإدخال */
    .section-header { font-size: 24px; font-weight: 700; color: #333333; margin-bottom: 5px; margin-top: 15px; }
    .section-subheader { font-size: 14px; color: #666666; margin-bottom: 25px; }
    
    /* بطاقات مخرجات التقييم التلقائي الفوري */
    .result-card { padding: 25px; border-radius: 12px; text-align: center; font-size: 23px; font-weight: 800; margin-top: 15px; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.04); }
    .status-irrigate { background-color: #FFF3CD; color: #856404; border: 1px solid #FFEBAA; }
    .status-safe { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
    
    /* بطاقة التوصيات الخاصة بالمحاصيل */
    .crop-recommendation { padding: 20px; border-radius: 12px; margin-top: 20px; background-color: #E3F2FD; color: #0D47A1; }
    .crop-recommendation.rtl { border-right: 5px solid #1E88E5; }
    .crop-recommendation.ltr { border-left: 5px solid #1E88E5; }
    .crop-title { font-size: 17px; font-weight: bold; margin-bottom: 10px; }
    
    /* التنسيق المفرود والنظيف تماماً للمصادر بأسفل واجهات الإدخال */
    .footer-title { font-size: 16px; font-weight: bold; color: #1B5E20; margin-top: 40px; margin-bottom: 12px; text-transform: uppercase; }
    .footer-text { font-size: 14px; color: #444444; margin-bottom: 8px; line-height: 1.5; }
    
    /* الكود الحاسم لإجبار الأرقام على الظهور بالصيغة الإنجليزية دائماً ومنع تحويل الويندوز لها */
    input, .stMetric, .result-card, .crop-recommendation, div, span, p { 
        font-variant-numeric: ltr-mapping !important; 
        font-feature-settings: "lnum" 1, "tnum" 1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================
# 2. تحميل ملفات الذكاء الاصطناعي وقاعدة بيانات المحاصيل الافتراضية
# ====================================================================
@st.cache_resource
def load_ai_assets():
    with open('smart_irrigation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

ai_model, ai_scaler = load_ai_assets()

crop_db = {
    "ar": {
        "🌱 بقوليات": {
            "فول مصري": {"temp": 18.0, "hum": 45.0, "wind": 8.0, "solar": 400.0, "soil": 35.0, "base_hours": 1.5},
            "فاصوليا بيضاء": {"temp": 22.0, "hum": 50.0, "wind": 10.0, "solar": 450.0, "soil": 40.0, "base_hours": 2.0},
            "بازلاء": {"temp": 16.0, "hum": 40.0, "wind": 7.0, "solar": 380.0, "soil": 38.0, "base_hours": 1.5}
        },
        "🍅 خضروات": {
            "طماطم حقلية": {"temp": 25.0, "hum": 60.0, "wind": 12.0, "solar": 550.0, "soil": 45.0, "base_hours": 2.5},
            "بطاطس": {"temp": 20.0, "hum": 55.0, "wind": 9.0, "solar": 500.0, "soil": 50.0, "base_hours": 2.5},
            "خيار": {"temp": 27.0, "hum": 65.0, "wind": 11.0, "solar": 600.0, "soil": 42.0, "base_hours": 3.0}
        },
        "🍊 فواكه": {
            "حمضيات": {"temp": 24.0, "hum": 55.0, "wind": 12.0, "solar": 500.0, "soil": 45.0, "base_hours": 3.5},
            "عنب": {"temp": 26.0, "hum": 40.0, "wind": 14.0, "solar": 650.0, "soil": 35.0, "base_hours": 3.0},
            "رمان": {"temp": 22.0, "hum": 35.0, "wind": 15.0, "solar": 580.0, "soil": 30.0, "base_hours": 3.5}
        },
        "🌽 حبوب وذرة": {
            "ذرة شامية": {"temp": 28.0, "hum": 50.0, "wind": 13.0, "solar": 600.0, "soil": 40.0, "base_hours": 3.0},
            "ذرة رفيعة": {"temp": 30.0, "hum": 35.0, "wind": 15.0, "solar": 700.0, "soil": 28.0, "base_hours": 2.0}
        }
    },
    "en": {
        "🌱 Legumes": {
            "Fava Beans": {"temp": 18.0, "hum": 45.0, "wind": 8.0, "solar": 400.0, "soil": 35.0, "base_hours": 1.5},
            "White Beans": {"temp": 22.0, "hum": 50.0, "wind": 10.0, "solar": 450.0, "soil": 40.0, "base_hours": 2.0},
            "Green Peas": {"temp": 16.0, "hum": 40.0, "wind": 7.0, "solar": 380.0, "soil": 38.0, "base_hours": 1.5}
        },
        "🍅 Vegetables": {
            "Tomatoes": {"temp": 25.0, "hum": 60.0, "wind": 12.0, "solar": 550.0, "soil": 45.0, "base_hours": 2.5},
            "Potatoes": {"temp": 20.0, "hum": 55.0, "wind": 9.0, "solar": 500.0, "soil": 50.0, "base_hours": 2.5},
            "Cucumbers": {"temp": 27.0, "hum": 65.0, "wind": 11.0, "solar": 600.0, "soil": 42.0, "base_hours": 3.0}
        },
        "🍊 Fruits": {
            "Citrus Fruits": {"temp": 24.0, "hum": 55.0, "wind": 12.0, "solar": 500.0, "soil": 45.0, "base_hours": 3.5},
            "Grapes": {"temp": 26.0, "hum": 40.0, "wind": 14.0, "solar": 650.0, "soil": 35.0, "base_hours": 3.0},
            "Pomegranates": {"temp": 22.0, "hum": 35.0, "wind": 15.0, "solar": 580.0, "soil": 30.0, "base_hours": 3.5}
        },
        "🌽 Grains & Maize": {
            "Corn / Maize": {"temp": 28.0, "hum": 50.0, "wind": 13.0, "solar": 600.0, "soil": 40.0, "base_hours": 3.0},
            "Sorghum": {"temp": 30.0, "hum": 35.0, "wind": 15.0, "solar": 700.0, "soil": 28.0, "base_hours": 2.0}
        }
    }
}

if "page" not in st.session_state:
    st.session_state.page = "main"

# ====================================================================
# 🗂️ الواجهة الأولى (الرئيسية): اختيار اللغة والأسماء المفتوحة بالمنتصف فقط
# ====================================================================
if st.session_state.page == "main":
    st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">🌱 IrrigationCare AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Smart Irrigation Predictive System</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="team-line-main">👥 Prepared by: Nada Rezqan | Al-Batoul Al-Zain | Sarah Al-Ahjri</div>', unsafe_allow_html=True)
    st.markdown('<div class="supervisor-line-main">🎓 Supervised by: Dr. Safwan Al-Shaibani</div>', unsafe_allow_html=True)
    
    st.write("---")
    
    lang_choice = st.selectbox("🌐 Select Interface Language / اختر لغة الواجهة:", ["-- Choose / اختر --", "العربية", "English"])
    
    if lang_choice == "العربية":
        st.session_state.page = "arabic_panel"
        st.rerun()
    elif lang_choice == "English":
        st.session_state.page = "english_panel"
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
# ====================================================================
# 🗂️ الواجهة الثانية: لوحة التحكم باللغة العربية
# ====================================================================
elif st.session_state.page == "arabic_panel":
    if st.button("◄ العودة للصفحة الرئيسية"):
        st.session_state.page = "main"
        st.rerun()
        
    st.markdown('<div style="direction: rtl; text-align: right;">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌾 المعايير البيئية ورطوبة التربة</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">يرجى تحديد تصنيف ونوع المحصول لمعاينة البيانات الحقلية تلقائياً.</div>', unsafe_allow_html=True)
    
    db_ar = crop_db["ar"]
    col_c1, col_c2 = st.columns(2)
    with col_c1: cat_selected = st.selectbox("اختر تصنيف المحصول الأساسي:", list(db_ar.keys()))
    with col_c2: crop_selected = st.selectbox("اختر المحصول الفرعي المحدد:", list(db_ar[cat_selected].keys()))
    
    defaults = db_ar[cat_selected][crop_selected]
    
    st.write("---")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1: temperature = st.number_input("درجة الحرارة الحالية (°م)", min_value=0.0, max_value=50.0, value=defaults["temp"], step=0.1, key="temp_ar")
    with row1_col2: humidity = st.number_input("رطوبة الهواء النسبية (%)", min_value=0.0, max_value=100.0, value=defaults["hum"], step=0.1, key="hum_ar")
    
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1: wind_speed = st.number_input("سرعة الرياح الحقلية (كم/ساعة)", min_value=0.0, max_value=50.0, value=defaults["wind"], step=0.1, key="wind_ar")
    with row2_col2: solar_radiation = st.number_input("كثافة الإشعاع الشمسي (واط/متر²)", min_value=0.0, max_value=1000.0, value=defaults["solar"], step=1.0, key="solar_ar")
    
    soil_moisture = st.number_input("نسبة رطوبة التربة الحالية (%)", min_value=0.0, max_value=100.0, value=defaults["soil"], step=0.1, key="soil_ar")
    
    input_data = np.array([[temperature, humidity, wind_speed, solar_radiation, soil_moisture]])
    input_scaled = ai_scaler.transform(input_data)
    prediction = ai_model.predict(input_scaled)
    
    if prediction == 0:
        calculated_hours = 0.0
    else:
        moisture_deficit = max(0.0, defaults["soil"] - soil_moisture)
        weather_factor = (temperature / defaults["temp"]) * (solar_radiation / defaults["solar"] if defaults["solar"] > 0 else 1.0)
        calculated_hours = round(defaults["base_hours"] + (moisture_deficit * 0.05) * weather_factor, 1)
        calculated_hours = min(6.0, max(1.0, calculated_hours))
    
    st.write("")
    st.markdown('<h4>📊 قرار تقييم خطر الري الفوري</h4>', unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(f'<div class="result-card status-irrigate">⚠️ تنبيه: الحقل بحاجة للري الفوري الآن<br><span style="font-size: 14px; font-weight: normal; color: #666; display: block; margin-top: 5px;">مستوى رطوبة التربة الحالي منخفض وغير آمن لاستدامة محصول ({crop_selected}).</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-card status-safe">✅ حالة التربة آمنة - لا حاجة لتشغيل الري<br><span style="font-size: 14px; font-weight: normal; color: #555; display: block; margin-top: 5px;">مخرجات الموديل تؤكد استقرار المحتوى المائي للتربة مع احتياج محصول ({crop_selected}).</span></div>', unsafe_allow_html=True)
        
    st.markdown(f"""
        <div class="crop-recommendation rtl">
            <div class="crop-title">💡 الإرشادات الفنية الخاصة بـ ({crop_selected}):</div>
            • <b>ساعات الري المطلوبة عند الجفاف بناءً على مدخلاتك الحالية:</b> يُنصح بالري المستمر لمدة <b>{calculated_hours} ساعة</b> لتغطية احتياج الجذور.<br>
            • <b>درجة الحرارة المرجعية للنمو:</b> يفضل استقرار الأجواء حول {defaults["temp"]}°م لتجنب الإجهاد المناخي.<br>
            • <b>الرطوبة الجوية المستهدفة:</b> النطاق الطبيعي الأفضل للمحصول هو {defaults["hum"]}%.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer-title">📚 مصادر وتوثيق النظام العلمي البرمجي</div>
        <div class="footer-text">• <b>Yemen Climate Context:</b> Calibrated using United Nations UNDP Historical Climate Reports [historical_weather].</div>
        <div class="footer-text">• <b>Algorithm & Core Models:</b> Trained via Scikit-Learn RandomForestClassifier & StandardScaler Pipeline.</div>
        <div class="footer-text">• <b>System Architecture:</b> Developed using Python Streamlit Advanced Component Layout Engine.</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# 🗂️ الواجهة الثالثة: لوحة التحكم باللغة الإنجليزية الكاملة مع التعليقات العربية
# ====================================================================
elif st.session_state.page == "english_panel":
    if st.button("◄ Return to Main Page"):
        st.session_state.page = "main"
        st.rerun()
        
    st.markdown('<div class="section-header">🌾 Environmental & Soil Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Please provide the localized field telemetry measurements below.</div>', unsafe_allow_html=True)
    
    db_en = crop_db["en"]
    col_c1, col_c2 = st.columns(2)
    with col_c1: cat_selected = st.selectbox("Select Core Crop Category:", list(db_en.keys()))
    with col_c2: crop_selected = st.selectbox("Select Target Specific Crop:", list(db_en[cat_selected].keys()))
    
    defaults = db_en[cat_selected][crop_selected]
    
    st.write("---")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1: temperature = st.number_input("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=defaults["temp"], step=0.1, key="temp_en")
    with row1_col2: humidity = st.number_input("Relative Air Humidity (%)", min_value=0.0, max_value=100.0, value=defaults["hum"], step=0.1, key="hum_en")
    
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1: wind_speed = st.number_input("Average Wind Speed (Km/h)", min_value=0.0, max_value=50.0, value=defaults["wind"], step=0.1, key="wind_en")
    with row2_col2: solar_radiation = st.number_input("Solar Radiation Intensity (W/m²)", min_value=0.0, max_value=1000.0, value=defaults["solar"], step=1.0, key="solar_en")
    
    soil_moisture = st.number_input("Current Soil Moisture Level (%)", min_value=0.0, max_value=100.0, value=defaults["soil"], step=0.1, key="soil_en")
    
    input_data = np.array([[temperature, humidity, wind_speed, solar_radiation, soil_moisture]])
    input_scaled = ai_scaler.transform(input_data)
    prediction = ai_model.predict(input_scaled)
    
    # بداية فحص التنبؤ الفوري لقرار الري في الواجهة الإنجليزية
    if prediction == 0:
        # إذا كانت التربة مكتفية بالماء، فإن ساعات الري المطلوبة تكون 0 ساعة تلقائياً
        calculated_hours = 0.0
    else:
        # إذا كانت الأرض جافة، يحسب الذكاء الاصطناعي الساعات حركياً بناءً على نسبة النقص في رطوبة التربة
        moisture_deficit = max(0.0, defaults["soil"] - soil_moisture)
        # إدخال تأثير الحرارة الحالية وقوة الإشعاع الشمسي لرفع الساعات بشكل منطقي ومحاكات الواقع
        weather_factor = (temperature / defaults["temp"]) * (solar_radiation / defaults["solar"] if defaults["solar"] > 0 else 1.0)
        # دمج العوامل في المعادلة الذكية وحساب الناتج وتقريبه لأقرب جزء من عشرة
        calculated_hours = round(defaults["base_hours"] + (moisture_deficit * 0.05) * weather_factor, 1)
        # وضع حد أقصى للري (6 ساعات) لحماية جذور النبات من التعفن
        calculated_hours = min(6.0, max(1.0, calculated_hours))
    
    st.write("")
    st.markdown('<h4>📊 Real-Time AI Inference Output</h4>', unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(f'<div class="result-card status-irrigate">⚠️ WARNING: IRRIGATION SYSTEM ACTIVATION REQUIRED<br><span style="font-size: 14px; font-weight: normal; color: #666; display: block; margin-top: 5px;">The soil moisture retention level is critically low for sustaining ({crop_selected}) under current parameters.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-card status-safe">✅ FIELD STATUS OPTIMAL - NO IRRIGATION NEEDED<br><span style="font-size: 14px; font-weight: normal; color: #555; display: block; margin-top: 5px;">The AI model verifies that the current volumetric soil moisture is fully adequate for ({crop_selected}).</span></div>', unsafe_allow_html=True)
        
    st.markdown(f"""
        <div class="crop-recommendation ltr">
            <div class="crop-title">💡 Technical Guidelines Calculated for ({crop_selected}):</div>
            • <b>Target Watering Duration Based On Your Inputs:</b> Requires exactly <b style="color: #0D47A1; font-size: 19px;">{calculated_hours} Hours</b> of optimized flow soaking.<br>
            • <b>Reference Growth Temperature Threshold:</b> Performs best around {defaults["temp"]}°C to avoid climatic stress.<br>
            • <b>Target Atmospheric Humidity Equilibrium:</b> Ideal moisture threshold sits at {defaults["hum"]}%.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer-title">📚 Project System Documentation & Sources</div>
        <div class="footer-text">• <b>Yemen Climate Context:</b> Calibrated using United Nations UNDP Historical Climate Reports [historical_weather].</div>
        <div class="footer-text">• <b>Algorithm & Core Models:</b> Trained via Scikit-Learn RandomForestClassifier & StandardScaler Pipeline.</div>
        <div class="footer-text">• <b>System Architecture:</b> Developed using Python Streamlit Advanced Component Layout Engine.</div>
    """, unsafe_allow_html=True)

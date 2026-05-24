import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import gdown

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
MODEL_PATH    = 'modelo_piel.onnx'
DRIVE_FILE_ID = '1HSo1z4Cvjw_bxKY6YHwhEma0qFHgRBHC'
IMAGE_SIZE    = 299
CLASES        = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

INFO_CLASES = {
    'akiec': {
        'nombre': 'Queratosis Actínica / Carcinoma Intraepitelial',
        'descripcion': 'Lesión precancerosa causada por exposición prolongada al sol. Puede evolucionar a carcinoma de células escamosas si no se trata a tiempo.',
        'urgencia': '🟡 Moderada', 'color': '#f59e0b',
        'recomendacion': 'Consultar a un dermatólogo en las próximas semanas para evaluación y posible biopsia.',
        'sintomas': 'Parches rugosos y escamosos, coloración rojiza o marrón, picazón o ardor leve.',
        'causas': 'Exposición crónica a rayos UV, piel clara, edad avanzada.',
        'tratamiento': 'Criocirugía, cremas tópicas (5-fluorouracilo, imiquimod), terapia fotodinámica.'
    },
    'bcc': {
        'nombre': 'Carcinoma Basocelular',
        'descripcion': 'El tipo más común de cáncer de piel. Crece lentamente y rara vez se dispersa a otras partes del cuerpo.',
        'urgencia': '🟠 Alta', 'color': '#f97316',
        'recomendacion': 'Consultar a un dermatólogo pronto. Tiene excelente pronóstico con tratamiento oportuno.',
        'sintomas': 'Bulto perlado o ceroso, lesión plana color carne, úlcera que no sana.',
        'causas': 'Exposición al sol, rayos UV artificiales, piel clara, antecedentes familiares.',
        'tratamiento': 'Cirugía de Mohs, escisión quirúrgica, radioterapia, terapia tópica.'
    },
    'bkl': {
        'nombre': 'Lesión Queratósica Benigna',
        'descripcion': 'Agrupa queratosis seborreica, lentigo solar y queratosis liquenoide benigna. Son completamente benignas.',
        'urgencia': '🟢 Baja', 'color': '#22c55e',
        'recomendacion': 'Monitorear cambios en tamaño, color o forma. Consultar si hay cambios notables.',
        'sintomas': 'Manchas marrones o negras de apariencia cerosa, bordes bien definidos.',
        'causas': 'Envejecimiento, exposición solar acumulada, factores genéticos.',
        'tratamiento': 'Generalmente no requiere tratamiento. Criocirugía si causa molestias estéticas.'
    },
    'df': {
        'nombre': 'Dermatofibroma',
        'descripcion': 'Tumor benigno de la piel, muy común. Suele ser firme al tacto y de color marrón rosado.',
        'urgencia': '🟢 Baja', 'color': '#22c55e',
        'recomendacion': 'Generalmente no requiere tratamiento. Consultar si causa molestias o crece rápidamente.',
        'sintomas': 'Nódulo firme, color marrón rosado, sensible al tacto, hoyuelo al pellizcar.',
        'causas': 'Posible reacción a picaduras de insectos o lesiones menores.',
        'tratamiento': 'No requiere tratamiento. Escisión quirúrgica si molesta.'
    },
    'mel': {
        'nombre': 'Melanoma',
        'descripcion': 'El tipo más peligroso de cáncer de piel. Se origina en los melanocitos y puede diseminarse rápidamente.',
        'urgencia': '🔴 Muy alta', 'color': '#ef4444',
        'recomendacion': '⚠️ CONSULTAR A UN DERMATÓLOGO URGENTEMENTE. El diagnóstico temprano es crítico.',
        'sintomas': 'Lunar asimétrico, bordes irregulares, múltiples colores, diámetro >6mm, evolución rápida.',
        'causas': 'Exposición UV, lunares atípicos, piel clara, antecedentes familiares.',
        'tratamiento': 'Cirugía, inmunoterapia, terapia dirigida, radioterapia según estadio.'
    },
    'nv': {
        'nombre': 'Nevo Melanocítico (Lunar)',
        'descripcion': 'Lunar común benigno formado por melanocitos agrupados. La gran mayoría son completamente inofensivos.',
        'urgencia': '🟢 Baja', 'color': '#22c55e',
        'recomendacion': 'Monitorear con la regla ABCDE. Consultar si hay cambios en asimetría, borde, color o diámetro.',
        'sintomas': 'Mancha o bulto de color uniforme (marrón, negro, rosado), bordes regulares.',
        'causas': 'Agrupamiento de melanocitos, factores genéticos, exposición solar en infancia.',
        'tratamiento': 'No requiere tratamiento salvo cambios sospechosos.'
    },
    'vasc': {
        'nombre': 'Lesión Vascular',
        'descripcion': 'Incluye angiomas, hemangiomas y otras lesiones de origen vascular. Generalmente benignas.',
        'urgencia': '🟢 Baja', 'color': '#22c55e',
        'recomendacion': 'Consultar si la lesión sangra, crece rápidamente o cambia de apariencia.',
        'sintomas': 'Coloración roja, violeta o azulada, puede ser plana o elevada.',
        'causas': 'Anomalías en vasos sanguíneos, factores congénitos o adquiridos.',
        'tratamiento': 'Láser, escleroterapia o cirugía si hay indicación médica o estética.'
    }
}

# ─────────────────────────────────────────────
# CARGAR MODELO ONNX
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Descargando modelo..."):
            gdown.download(f'https://drive.google.com/uc?id={DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
    return ort.InferenceSession(MODEL_PATH)


def preprocesar_imagen(img):
    img = img.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32)
    # Preprocesamiento InceptionResNetV2
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def predecir(session, img):
    x = preprocesar_imagen(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    probs = outputs[0][0]
    idx = int(np.argmax(probs))
    return CLASES[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────
# CHAT INTELIGENTE
# ─────────────────────────────────────────────
def consultar_agente(pregunta, ctx):
    p = pregunta.lower()
    nombre   = ctx['nombre']
    sint     = ctx['sintomas']
    causas   = ctx['causas']
    trat     = ctx['tratamiento']
    rec      = ctx['recomendacion']
    urgencia = ctx['urgencia']
    confianza = ctx['confianza']
    codigo   = ctx['codigo']
    desc     = ctx['descripcion']
    disclaimer = "\n\n⚠️ *Recuerda: este análisis es académico y no reemplaza la consulta con un dermatólogo certificado.*"

    # ── Saludos
    if any(w in p for w in ["hola", "buenas", "buenos", "hi", "hey", "saludos", "buen día", "buenas tardes", "buenas noches"]):
        return f"¡Hola! 👋 Soy DermAgent, tu asistente de análisis de lesiones cutáneas. Ya analicé la imagen y detecté **{nombre}** con una confianza del **{confianza*100:.1f}%**. Puedes preguntarme sobre síntomas, causas, tratamiento o urgencia médica. ¿En qué te puedo ayudar?"

    # ── Gracias
    elif any(w in p for w in ["gracias", "thank", "perfecto", "genial", "excelente", "ok", "entiendo", "entendido"]):
        return f"¡Con gusto! 😊 Recuerda siempre confirmar con un dermatólogo certificado. ¿Tienes alguna otra pregunta sobre el **{nombre}**?"

    # ── Ayuda
    elif any(w in p for w in ["ayuda", "help", "qué haces", "para qué", "cómo funciona"]):
        return f"Soy DermAgent 🔬, especializado en lesiones cutáneas. Puedo ayudarte con síntomas, causas, tratamiento, urgencia y prevención. Analicé **{nombre}**. ¿Qué quieres saber?"

    # ── Peligro
    elif any(w in p for w in ["peligro", "grave", 'cancer', 'cáncer', 'maligno', 'malo', 'mortal', 'muero', 'mori']):
        if codigo == 'mel':
            return f"El **{nombre}** es la forma más peligrosa de cáncer de piel. Sin embargo, detectado a tiempo tiene muy buen pronóstico. Es fundamental que consultes a un dermatólogo lo antes posible.{disclaimer}"
        elif codigo in ['bcc', 'akiec']:
            return f"El **{nombre}** requiere atención médica, pero con tratamiento oportuno tiene excelente pronóstico. {rec}{disclaimer}"
        else:
            return f"El **{nombre}** es una lesión **benigna**, no es cancerosa ni peligrosa. {rec}{disclaimer}"

    elif any(w in p for w in ['hacer', 'hago', 'debo', 'recomien', 'consejo', 'pasos']):
        return f"Para el **{nombre}**: {rec}\n\n- Fotografiar la lesión para detectar cambios\n- Evitar exposición solar directa\n- No manipular ni rascar la lesión{disclaimer}"

    elif any(w in p for w in ['síntoma', 'sintoma', 'duele', 'dolor', 'pica', 'ardor', 'molest', 'aspecto']):
        return f"Los síntomas típicos del **{nombre}** incluyen: {sint}{disclaimer}"

    elif any(w in p for w in ['causa', 'origen', 'por qué', 'porque', 'produce', 'aparece', 'sale']):
        return f"El **{nombre}** generalmente aparece por: {causas}{disclaimer}"

    elif any(w in p for w in ['tratamiento', 'cura', 'eliminar', 'quitar', 'operar', 'medicina', 'crema']):
        return f"Para el **{nombre}**, las opciones de tratamiento incluyen: {trat}{disclaimer}"

    elif any(w in p for w in ['seguro', 'confianza', 'precis', 'exacto', 'probabilidad', 'modelo']):
        nivel = 'muy seguro' if confianza > 0.85 else 'moderadamente seguro' if confianza > 0.6 else 'poco seguro'
        return f"El modelo clasificó esta imagen como **{nombre}** con una confianza del **{confianza*100:.1f}%** — está {nivel}.{disclaimer}"

    elif any(w in p for w in ['urgent', 'médico', 'doctor', 'dermatólogo', 'hospital', 'cuándo', 'cuando']):
        if codigo in ['mel', 'bcc', 'akiec']:
            return f"Con **{nombre}**, urgencia **{urgencia}**. Busca cita con un dermatólogo lo antes posible.{disclaimer}"
        else:
            return f"El **{nombre}** tiene urgencia **{urgencia}**. No es emergencia pero confirma con un dermatólogo.{disclaimer}"

    elif any(w in p for w in ['prevenir', 'evitar', 'proteger', 'sol', 'bloqueador']):
        return f"Para prevenir lesiones cutáneas:\n- Bloqueador solar SPF 50+ diario\n- Evitar el sol entre 10am y 4pm\n- Ropa protectora y sombrero\n- Revisar tu piel mensualmente (regla ABCDE)\n- Visitar al dermatólogo anualmente{disclaimer}"

    elif any(w in p for w in ['contagia', 'contagioso', 'pega', 'transmite']):
        return f"No, el **{nombre}** no es contagioso.{disclaimer}"

    elif any(w in p for w in ['qué es', 'que es', 'explica', 'descripción', 'informa', 'cuéntame', 'dime']):
        return f"El **{nombre}** es: {desc}\n\n{rec}{disclaimer}"

    else:
        return f"Sobre el **{nombre}**:\n\n📋 {desc}\n\n🔬 Síntomas: {sint}\n\n✅ {rec}{disclaimer}"


# ─────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────
st.set_page_config(page_title="DermAgent", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0f1117; color: #e8e8e8; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }
    .header-box { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border: 1px solid #2a3050; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; }
    .resultado-card { border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1rem; border-left: 5px solid; background: #1a1f2e; }
    .disclaimer { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 10px; padding: 0.8rem 1.2rem; font-size: 0.8rem; color: #888; margin-top: 1.5rem; }
    .chat-msg-user { background: #1e3a5f; border-radius: 12px 12px 2px 12px; padding: 0.7rem 1rem; margin: 0.4rem 0; max-width: 85%; margin-left: auto; font-size: 0.9rem; }
    .chat-msg-bot { background: #1a1f2e; border: 1px solid #2a3050; border-radius: 12px 12px 12px 2px; padding: 0.7rem 1rem; margin: 0.4rem 0; max-width: 90%; font-size: 0.9rem; }
    .stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; border: none; border-radius: 10px; padding: 0.5rem 1.5rem; font-weight: 500; width: 100%; }
</style>
""", unsafe_allow_html=True)

if 'resultado' not in st.session_state:
    st.session_state.resultado = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

st.markdown("""
<div class="header-box">
    <h1 style="margin:0; font-size:2rem;">🔬 DermAgent</h1>
    <p style="margin:0.3rem 0 0; color:#9ca3af;">Asistente inteligente de clasificación de lesiones cutáneas · IRV2 + Soft Attention</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1.2], gap="medium")

with col1:
    st.markdown("#### 📷 Imagen")
    archivo = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
    if archivo:
        img = Image.open(archivo)
        st.image(img, width=280)
        if st.button("🔍 Analizar"):
            with st.spinner("Analizando..."):
                try:
                    session = cargar_modelo()
                    clase, confianza, probs = predecir(session, img)
                    info = INFO_CLASES[clase]
                    st.session_state.resultado = {
                        'clase': clase, 'confianza': confianza, 'probs': probs, 'info': info,
                        'codigo': clase, 'nombre': info['nombre'], 'descripcion': info['descripcion'],
                        'urgencia': info['urgencia'], 'sintomas': info['sintomas'],
                        'causas': info['causas'], 'tratamiento': info['tratamiento'],
                        'recomendacion': info['recomendacion']
                    }
                    st.session_state.chat_messages = []
                    msg = f"He analizado la imagen. El modelo detectó **{info['nombre']}** con una confianza del **{confianza*100:.1f}%**. ¿Tienes alguna pregunta?"
                    st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Sube una foto de la lesión para comenzar.")

with col2:
    st.markdown("#### 📊 Resultado")
    if st.session_state.resultado:
        r = st.session_state.resultado
        info = r['info']
        color = info['color']
        st.markdown(f"""
        <div class="resultado-card" style="border-color:{color};">
            <p style="color:{color}; font-weight:600; font-size:0.8rem; margin:0;">DIAGNÓSTICO SUGERIDO</p>
            <h3 style="margin:0.3rem 0; font-size:1.2rem;">{info['nombre']}</h3>
            <p style="color:#9ca3af; margin:0; font-size:0.85rem;">Confianza: <strong style="color:white;">{r['confianza']*100:.1f}%</strong> &nbsp;·&nbsp; {info['urgencia']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**📋 Descripción:** {info['descripcion']}")
        st.markdown(f"**💊 Tratamiento:** {info['tratamiento']}")
        st.markdown(f"**✅ Recomendación:** {info['recomendacion']}")
        st.markdown("---")
        st.markdown("**Probabilidades:**")
        for cls, prob in zip(CLASES, r['probs']):
            nombre_corto = INFO_CLASES[cls]['nombre'].split('/')[0].strip()[:22]
            color_barra = info['color'] if cls == r['clase'] else '#374151'
            pct = prob * 100
            st.markdown(f"""
            <div style="margin-bottom:1px;">
                <span style="font-size:0.75rem; color:#9ca3af;">{nombre_corto}</span>
                <span style="float:right; font-size:0.75rem; color:{'white' if cls==r['clase'] else '#6b7280'};">{pct:.1f}%</span>
            </div>
            <div style="background:#1f2937; border-radius:3px; height:6px; margin-bottom:8px;">
                <div style="background:{color_barra}; width:{pct}%; height:6px; border-radius:3px;"></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1a1f2e; border-radius:12px; padding:2rem; text-align:center; color:#4b5563;">
            <p style="font-size:2.5rem; margin:0;">🩺</p>
            <p style="margin:0.5rem 0 0;">El resultado aparecerá aquí</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("#### 💬 Pregúntale al asistente")
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div style="text-align:center; color:#4b5563; padding:2rem;">
                <p style="font-size:2rem;">🤖</p>
                <p style="font-size:0.85rem;">Analiza una imagen primero.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_messages:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-msg-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    if st.session_state.resultado:
        pregunta = st.chat_input("Escribe tu pregunta sobre la lesión...")
        if pregunta:
            st.session_state.chat_messages.append({"role": "user", "content": pregunta})
            respuesta = consultar_agente(pregunta, st.session_state.resultado)
            st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})
            st.rerun()
    else:
        st.chat_input("Analiza una imagen primero...", disabled=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Aviso:</strong> Herramienta académica basada en IA. <strong>No reemplaza el diagnóstico médico profesional.</strong>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ ¿Qué lesiones puede detectar?"):
    cols = st.columns(2)
    for i, (cls, info) in enumerate(INFO_CLASES.items()):
        with cols[i % 2]:
            st.markdown(f"**{cls.upper()} — {info['nombre']}**\n\n{info['descripcion']}\n\n{info['urgencia']}\n\n---")

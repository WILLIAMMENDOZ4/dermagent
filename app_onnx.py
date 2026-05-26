import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import gdown
from sentence_transformers import SentenceTransformer, util

MODEL_PATH    = 'modelo_piel.onnx'
DRIVE_FILE_ID = '1l1CjsAkzsdKjoShFMRbLFa4eMEgUHqbI'
IMAGE_SIZE    = 299
CLASES        = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

INFO_CLASES = {
    'akiec': {
        'nombre': 'Queratosis Actínica', 'codigo_display': 'AKIEC',
        'descripcion': 'Lesión precancerosa causada por exposición prolongada al sol. Puede evolucionar a carcinoma de células escamosas si no se trata a tiempo.',
        'urgencia': 'Moderada', 'urgencia_icon': '⚠️', 'color': '#d97706', 'color_bg': '#fef3c7', 'color_text': '#92400e',
        'recomendacion': 'Consultar a un dermatólogo en las próximas semanas para evaluación y posible biopsia.',
        'sintomas': 'Parches rugosos y escamosos, coloración rojiza o marrón, picazón o ardor leve.',
        'causas': 'Exposición crónica a rayos UV, piel clara, edad avanzada.',
        'tratamiento': 'Criocirugía, cremas tópicas (5-fluorouracilo, imiquimod), terapia fotodinámica.',
        'tipo': 'Precancerosa'
    },
    'bcc': {
        'nombre': 'Carcinoma Basocelular', 'codigo_display': 'BCC',
        'descripcion': 'El tipo más común de cáncer de piel. Crece lentamente y rara vez se dispersa, pero requiere tratamiento médico.',
        'urgencia': 'Alta', 'urgencia_icon': '🔶', 'color': '#ea580c', 'color_bg': '#fff7ed', 'color_text': '#9a3412',
        'recomendacion': 'Consultar a un dermatólogo pronto. Tiene excelente pronóstico con tratamiento oportuno.',
        'sintomas': 'Bulto perlado o ceroso, lesión plana color carne, úlcera que no sana.',
        'causas': 'Exposición solar acumulada, piel clara, antecedentes familiares, rayos UV artificiales.',
        'tratamiento': 'Cirugía de Mohs, escisión quirúrgica, radioterapia, terapia tópica.',
        'tipo': 'Maligna'
    },
    'bkl': {
        'nombre': 'Lesión Queratósica Benigna', 'codigo_display': 'BKL',
        'descripcion': 'Agrupa queratosis seborreica, lentigo solar y queratosis liquenoide. Son completamente benignas y no cancerosas.',
        'urgencia': 'Baja', 'urgencia_icon': '✅', 'color': '#16a34a', 'color_bg': '#f0fdf4', 'color_text': '#14532d',
        'recomendacion': 'Monitorear cambios en tamaño, color o forma. Consultar si hay cambios notables.',
        'sintomas': 'Manchas marrones o negras de apariencia cerosa, bordes bien definidos.',
        'causas': 'Envejecimiento natural, exposición solar acumulada, factores genéticos.',
        'tratamiento': 'Generalmente no requiere tratamiento. Criocirugía si causa molestias estéticas.',
        'tipo': 'Benigna'
    },
    'df': {
        'nombre': 'Dermatofibroma', 'codigo_display': 'DF',
        'descripcion': 'Tumor benigno de la piel muy común. Suele ser firme al tacto y de color marrón rosado.',
        'urgencia': 'Baja', 'urgencia_icon': '✅', 'color': '#16a34a', 'color_bg': '#f0fdf4', 'color_text': '#14532d',
        'recomendacion': 'No requiere tratamiento. Consultar si causa molestias o crece rápidamente.',
        'sintomas': 'Nódulo firme, color marrón rosado, sensible al tacto, hoyuelo al pellizcar.',
        'causas': 'Posible reacción a picaduras de insectos o lesiones menores. Más común en mujeres.',
        'tratamiento': 'No requiere tratamiento. Escisión quirúrgica solo si genera molestias.',
        'tipo': 'Benigna'
    },
    'mel': {
        'nombre': 'Melanoma', 'codigo_display': 'MEL',
        'descripcion': 'El tipo más peligroso de cáncer de piel. Se origina en los melanocitos y puede diseminarse a otros órganos.',
        'urgencia': 'Muy alta', 'urgencia_icon': '🚨', 'color': '#dc2626', 'color_bg': '#fef2f2', 'color_text': '#7f1d1d',
        'recomendacion': 'CONSULTAR A UN DERMATÓLOGO URGENTEMENTE. El diagnóstico y tratamiento temprano son críticos.',
        'sintomas': 'Lunar asimétrico, bordes irregulares, múltiples colores, diámetro >6mm, evolución rápida (ABCDE).',
        'causas': 'Exposición UV intensa, lunares atípicos, piel clara, antecedentes familiares, inmunosupresión.',
        'tratamiento': 'Cirugía, inmunoterapia, terapia dirigida, radioterapia según estadio.',
        'tipo': 'Maligna'
    },
    'nv': {
        'nombre': 'Nevo Melanocítico', 'codigo_display': 'NV',
        'descripcion': 'Lunar común benigno formado por melanocitos agrupados. La gran mayoría son completamente inofensivos.',
        'urgencia': 'Baja', 'urgencia_icon': '✅', 'color': '#16a34a', 'color_bg': '#f0fdf4', 'color_text': '#14532d',
        'recomendacion': 'Monitorear con la regla ABCDE. Consultar si hay cambios en asimetría, borde, color o diámetro.',
        'sintomas': 'Mancha o bulto de color uniforme (marrón, negro, rosado), bordes regulares y simétricos.',
        'causas': 'Agrupamiento de melanocitos, factores genéticos, exposición solar en la infancia.',
        'tratamiento': 'No requiere tratamiento salvo cambios sospechosos. Extirpación si hay duda diagnóstica.',
        'tipo': 'Benigna'
    },
    'vasc': {
        'nombre': 'Lesión Vascular', 'codigo_display': 'VASC',
        'descripcion': 'Incluye angiomas, hemangiomas y otras lesiones de origen vascular. Generalmente benignas.',
        'urgencia': 'Baja', 'urgencia_icon': '✅', 'color': '#16a34a', 'color_bg': '#f0fdf4', 'color_text': '#14532d',
        'recomendacion': 'Consultar si la lesión sangra, crece rápidamente o cambia de apariencia.',
        'sintomas': 'Coloración roja, violeta o azulada, puede ser plana o elevada, no desaparece al presionar.',
        'causas': 'Anomalías en vasos sanguíneos, factores congénitos o adquiridos.',
        'tratamiento': 'Láser, escleroterapia o cirugía si hay indicación médica o estética.',
        'tipo': 'Benigna'
    }
}

INTENCIONES = {
    'saludo': ["hola", "buenas", "buenos días", "buenas tardes", "hey", "saludos", "cómo estás", "qué tal"],
    'gracias': ["gracias", "muchas gracias", "perfecto", "genial", "excelente", "entendido", "ok gracias"],
    'ayuda': ["qué puedes hacer", "para qué sirves", "cómo funciona", "ayúdame", "qué información me das"],
    'peligro': ["es peligroso", "es grave", "tengo cáncer", "es maligno", "me voy a morir", "debo preocuparme"],
    'recomendacion': ["qué debo hacer", "qué hago ahora", "cuáles son los pasos", "qué me recomiendas", "cómo procedo"],
    'sintomas': ["cuáles son los síntomas", "cómo se ve", "qué aspecto tiene", "me duele", "siento picazón", "qué molestias"],
    'causas': ["por qué aparece", "cuál es la causa", "de dónde viene", "qué lo origina", "por qué me salió"],
    'tratamiento': ["cómo se trata", "tiene cura", "qué medicamento", "cómo me lo quitan", "hay crema", "se puede operar"],
    'urgencia': ["cuándo ir al médico", "es urgente", "puedo esperar", "qué tan rápido debo ir al dermatólogo"],
    'prevencion': ["cómo lo prevengo", "cómo evitar", "qué bloqueador usar", "debo evitar el sol", "cómo proteger mi piel"],
    'contagio': ["se contagia", "es contagioso", "lo puedo pegar", "se transmite", "es infeccioso"],
    'precision': ["qué tan preciso", "confías en el resultado", "cuánta confianza tiene", "puede estar equivocado"],
    'descripcion': ["qué es esto", "explícame qué es", "cuéntame sobre esta lesión", "dame información", "qué significa"]
}

@st.cache_resource
def cargar_nlp():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def clasificar_intencion(pregunta, nlp_model):
    emb_pregunta = nlp_model.encode(pregunta, convert_to_tensor=True)
    mejor_intencion = 'descripcion'
    mejor_score = -1
    for intencion, ejemplos in INTENCIONES.items():
        emb_ejemplos = nlp_model.encode(ejemplos, convert_to_tensor=True)
        scores = util.cos_sim(emb_pregunta, emb_ejemplos)
        score_max = float(scores.max())
        if score_max > mejor_score:
            mejor_score = score_max
            mejor_intencion = intencion
    return mejor_intencion, mejor_score

def consultar_agente(pregunta, ctx, nlp_model):
    nombre = ctx['nombre']; sint = ctx['sintomas']; causas = ctx['causas']
    trat = ctx['tratamiento']; rec = ctx['recomendacion']; urgencia = ctx['urgencia']
    confianza = ctx['confianza']; codigo = ctx['codigo']; desc = ctx['descripcion']
    d = "\n\n*⚕️ Este análisis es orientativo. Consulta siempre a un dermatólogo certificado.*"
    intencion, _ = clasificar_intencion(pregunta, nlp_model)

    if intencion == 'saludo':
        return f"¡Hola! Soy **DermAI**, tu asistente dermatológico. Acabo de analizar la imagen y detecté **{nombre}** con una confianza del **{confianza*100:.1f}%**.\n\nPuedes preguntarme sobre síntomas, causas, tratamiento, urgencia o prevención. ¿En qué te puedo ayudar?"
    elif intencion == 'gracias':
        return f"¡Con mucho gusto! Recuerda confirmar siempre con un dermatólogo. ¿Tienes otra pregunta sobre el **{nombre}**?"
    elif intencion == 'ayuda':
        return f"Soy **DermAI**, un asistente de diagnóstico dermatológico basado en IA.\n\nPuedo responderte sobre:\n- Descripción y tipo de la lesión\n- Síntomas y causas\n- Opciones de tratamiento\n- Nivel de urgencia médica\n- Prevención y cuidados\n\nActualmente analicé **{nombre}**. ¿Qué deseas saber?"
    elif intencion == 'peligro':
        if codigo == 'mel':
            return f"El **{nombre}** es la forma más peligrosa de cáncer de piel. Detectado a tiempo tiene mejor pronóstico, pero es crucial consultar a un dermatólogo **de inmediato**.{d}"
        elif codigo in ['bcc', 'akiec']:
            return f"El **{nombre}** requiere atención médica, pero **no es la lesión más agresiva**. Con tratamiento oportuno tiene excelente pronóstico.\n\n{rec}{d}"
        else:
            return f"Buenas noticias: el **{nombre}** es una lesión **benigna**, no es cancerosa. No hay motivo de alarma, aunque siempre es bueno hacer seguimiento periódico.\n\n{rec}{d}"
    elif intencion == 'recomendacion':
        return f"Para el **{nombre}**:\n\n1. {rec}\n2. Fotografía la lesión periódicamente para detectar cambios\n3. Evita la exposición solar directa sobre la zona\n4. No manipules ni rasques la lesión\n5. Lleva un registro de cambios en tamaño, color o forma{d}"
    elif intencion == 'sintomas':
        return f"Síntomas típicos del **{nombre}**:\n\n{sint}\n\nSi hay dolor intenso, sangrado espontáneo o crecimiento muy rápido, consulta a un médico de inmediato.{d}"
    elif intencion == 'causas':
        return f"El **{nombre}** generalmente aparece por:\n\n{causas}\n\nEl uso diario de bloqueador solar SPF 50+ es una de las mejores medidas preventivas.{d}"
    elif intencion == 'tratamiento':
        return f"Opciones de tratamiento para el **{nombre}**:\n\n{trat}\n\nEl tratamiento específico lo determina el dermatólogo según tamaño, ubicación y características de la lesión.{d}"
    elif intencion == 'urgencia':
        if codigo in ['mel', 'bcc', 'akiec']:
            return f"Con diagnóstico de **{nombre}** (urgencia: {urgencia}), busca cita con un dermatólogo **lo antes posible**, idealmente esta semana.{d}"
        else:
            return f"El **{nombre}** tiene urgencia {urgencia}. No es una emergencia inmediata, pero sí es recomendable una consulta dermatológica para confirmar el diagnóstico.{d}"
    elif intencion == 'prevencion':
        return f"Para prevenir lesiones cutáneas:\n\n- Bloqueador solar **SPF 50+** todos los días\n- Evitar el sol entre 10am y 4pm\n- Ropa protectora, sombrero y gafas UV\n- Revisión mensual de la piel con la **regla ABCDE**\n- Visita al dermatólogo al menos una vez al año{d}"
    elif intencion == 'contagio':
        return f"No, el **{nombre}** **no es contagioso**. No se transmite por contacto directo, aire ni ninguna otra vía.{d}"
    elif intencion == 'precision':
        nivel = 'muy alta' if confianza > 0.85 else 'moderada' if confianza > 0.6 else 'baja'
        return f"El modelo clasificó esta imagen como **{nombre}** con confianza del **{confianza*100:.1f}%** (precisión {nivel}).\n\nEl sistema usa **InceptionResNetV2 + Soft Attention** entrenado con HAM10000 (~10.000 imágenes). Aún así, ningún modelo de IA reemplaza el diagnóstico clínico de un dermatólogo.{d}"
    else:
        return f"**{nombre}**: {desc}\n\n**Síntomas:** {sint}\n\n**Recomendación:** {rec}{d}"

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo de clasificación..."):
            gdown.download(f'https://drive.google.com/uc?id={DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
    return ort.InferenceSession(MODEL_PATH)

def preprocesar_imagen(img):
    img = img.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def predecir(session, img):
    x = preprocesar_imagen(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    probs = outputs[0][0]
    idx = int(np.argmax(probs))
    return CLASES[idx], float(probs[idx]), probs

st.set_page_config(page_title="DermAI", page_icon="🩺", layout="wide")

st.markdown("""
<div class="header">
    <div class="header-logo">🩺</div>
    <div>
        <p class="header-title">DermAI — Asistente Dermatológico Inteligente</p>
        <p class="header-sub">
            Clasificación de lesiones cutáneas &nbsp;·&nbsp; IRV2 + Soft Attention &nbsp;·&nbsp; NLP Semántico
            &nbsp;&nbsp;
            <span class="badge badge-blue">HAM10000</span>
            &nbsp;
            <span class="badge badge-green">7 tipos de lesiones</span>
            &nbsp;

# ─────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────
st.set_page_config(page_title="DermAI", page_icon="🩺", layout="wide")

st.markdown("""
<style>

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f8fafc; color: #1e293b; }
.header { background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 1rem 2rem; display: flex; align-items: center; gap: 16px; margin-bottom: 1.5rem; }
.header-logo { width: 42px; height: 42px; background: #0f172a; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; }
.header-title { font-size: 1.3rem; font-weight: 600; color: #0f172a; margin: 0; }
.header-sub { font-size: 0.78rem; color: #64748b; margin: 0; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 500; }
.badge-blue { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.badge-green { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
.card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; }
.card-title { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.stat-row { display: flex; gap: 8px; margin: 0.8rem 0; }
.stat-box { flex: 1; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem; text-align: center; }
.stat-val { font-size: 1.1rem; font-weight: 600; color: #0f172a; }
.stat-lbl { font-size: 0.7rem; color: #94a3b8; }
.info-row { display: flex; gap: 8px; margin: 0.4rem 0; font-size: 0.85rem; }
.info-lbl { color: #94a3b8; font-weight: 600; min-width: 100px; }
.info-val { color: #e2e8f0; flex: 1; }
.prob-bar { height: 5px; background: #e2e8f0; border-radius: 3px; margin: 3px 0 8px; }
.prob-fill { height: 5px; border-radius: 3px; }
.chat-container { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; min-height: 380px; max-height: 380px; overflow-y: auto; }
.msg-user { background: #0f172a; color: white; border-radius: 16px 16px 4px 16px; padding: 0.6rem 1rem; margin: 0.4rem 0 0.4rem auto; max-width: 80%; font-size: 0.88rem; display: table; margin-left: auto; }
.msg-bot { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px 16px 16px 4px; padding: 0.6rem 1rem; margin: 0.4rem 0; max-width: 88%; font-size: 0.88rem; color: #1e293b; }
.disclaimer { background: #fefce8; border: 1px solid #fde68a; border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.78rem; color: #713f12; margin-top: 1rem; }
.abcde-card { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.8rem; }
.abcde-title { font-size: 0.75rem; font-weight: 600; color: #0369a1; margin-bottom: 0.5rem; }
.abcde-row { display: flex; gap: 6px; }
.abcde-item { flex: 1; background: white; border: 1px solid #bae6fd; border-radius: 6px; padding: 0.4rem; text-align: center; }
.abcde-letter { font-size: 1rem; font-weight: 700; color: #0284c7; }
.abcde-word { font-size: 0.62rem; color: #64748b; }
.stButton > button { background: #0f172a !important; color: white !important; border: 2px solid #334155 !important; border-radius: 8px !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; width: 100% !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

if 'resultado' not in st.session_state:
    st.session_state.resultado = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'modal_cerrado' not in st.session_state:
    st.session_state.modal_cerrado = False

# ── Modal de bienvenida ──────────────────────
if not st.session_state.modal_cerrado:
    st.markdown("""
    <div style="position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.75); z-index:9999; display:flex; align-items:center; justify-content:center;">
        <div style="background:#ffffff; border-radius:16px; padding:2rem 2.5rem; max-width:520px; width:90%;">
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:1.2rem;">
                <div style="width:52px; height:52px; background:#0f172a; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:26px;">🩺</div>
                <div>
                    <p style="margin:0; font-size:1.25rem; font-weight:700; color:#0f172a;">Bienvenido a DermAI</p>
                    <p style="margin:0; font-size:0.82rem; color:#64748b;">Asistente Dermatológico Inteligente</p>
                </div>
            </div>
            <p style="color:#334155; font-size:0.9rem; line-height:1.65; margin-bottom:1rem;">
                <b>DermAI</b> es un sistema académico de apoyo diagnóstico que combina <b>visión artificial</b> y <b>procesamiento de lenguaje natural</b> para analizar lesiones cutáneas.
            </p>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1rem;">
                <p style="margin:0 0 0.8rem; font-size:0.82rem; font-weight:700; color:#0f172a;">¿Cómo usar DermAI?</p>
                <div style="display:flex; gap:10px; margin-bottom:8px; align-items:center;">
                    <span style="background:#0f172a; color:white; border-radius:50%; min-width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700;">1</span>
                    <span style="font-size:0.85rem; color:#334155;">Sube una imagen de la lesión cutánea</span>
                </div>
                <div style="display:flex; gap:10px; margin-bottom:8px; align-items:center;">
                    <span style="background:#0f172a; color:white; border-radius:50%; min-width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700;">2</span>
                    <span style="font-size:0.85rem; color:#334155;">El modelo IRV2 + Soft Attention clasifica entre 7 tipos de lesiones</span>
                </div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <span style="background:#0f172a; color:white; border-radius:50%; min-width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700;">3</span>
                    <span style="font-size:0.85rem; color:#334155;">El asistente NLP responde tus preguntas en lenguaje natural</span>
                </div>
            </div>
            <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                <p style="margin:0; font-size:0.8rem; color:#b91c1c;">⚕️ <b>Aviso médico:</b> Esta herramienta es de uso académico y no reemplaza el diagnóstico de un dermatólogo certificado.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✓ Entendido — Comenzar análisis", key="cerrar_modal"):
        st.session_state.modal_cerrado = True
        st.rerun()
    st.stop()

# ── Layout principal ─────────────────────────
col1, col2, col3 = st.columns([1, 1.1, 1.2], gap="medium")

with col1:
    st.markdown('<div class="card-title">📷 Imagen dermatológica</div>', unsafe_allow_html=True)
    archivo = st.file_uploader("Sube una imagen JPG o PNG de la lesión", type=['jpg','jpeg','png'], label_visibility='collapsed')
    if archivo:
        img = Image.open(archivo)
        st.image(img, use_column_width=True)
        if st.button("🔬 Analizar lesión"):
            with st.spinner("Procesando imagen con IRV2 + Soft Attention..."):
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
                    st.session_state.chat_messages.append({"role": "assistant",
                        "content": f"Análisis completado. El modelo detectó **{info['nombre']}** ({info['codigo_display']}) con una confianza del **{confianza*100:.1f}%**.\n\nPuedes preguntarme sobre síntomas, causas, tratamiento o urgencia. ¿En qué te puedo ayudar?"})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        st.markdown("""
        <div class="abcde-card">
            <div class="abcde-title">Regla ABCDE para lunares sospechosos</div>
            <div class="abcde-row">
                <div class="abcde-item"><div class="abcde-letter">A</div><div class="abcde-word">Asimetría</div></div>
                <div class="abcde-item"><div class="abcde-letter">B</div><div class="abcde-word">Borde</div></div>
                <div class="abcde-item"><div class="abcde-letter">C</div><div class="abcde-word">Color</div></div>
                <div class="abcde-item"><div class="abcde-letter">D</div><div class="abcde-word">Diámetro</div></div>
                <div class="abcde-item"><div class="abcde-letter">E</div><div class="abcde-word">Evolución</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f1f5f9; border:2px dashed #cbd5e1; border-radius:12px; padding:2.5rem; text-align:center; color:#94a3b8;">
            <div style="font-size:2.5rem;">🔬</div>
            <p style="margin:0.5rem 0 0; font-size:0.9rem;">Sube una imagen dermatológica<br>para iniciar el análisis</p>
            <p style="margin:0.5rem 0 0; font-size:0.75rem;">Formatos: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-title" style="text-transform:uppercase; letter-spacing:0.05em; font-size:0.72rem; color:#64748b; font-weight:600;">📋 Resultado del análisis</div>', unsafe_allow_html=True)
    if st.session_state.resultado:
        r = st.session_state.resultado
        info = r['info']
        color = info['color']
        color_bg = info['color_bg']
        color_text = info['color_text']
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {color};">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:0.72rem; font-weight:700; color:#94a3b8; letter-spacing:0.08em; text-transform:uppercase;">DIAGNÓSTICO SUGERIDO</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">{info['nombre']}</div>
                    <div style="font-size:0.82rem; color:{color};">{info['tipo']} &nbsp;·&nbsp; {info['urgencia_icon']} Urgencia {info['urgencia']}</div>
                </div>
                <div style="background:{color_bg}; color:{color_text}; border:1px solid {color}; border-radius:8px; padding:4px 12px; font-size:0.85rem; font-weight:700;">{info['codigo_display']}</div>
            </div>
            <div class="stat-row">
                <div class="stat-box"><div class="stat-val">{r['confianza']*100:.1f}%</div><div class="stat-lbl">Confianza</div></div>
                <div class="stat-box"><div class="stat-val">{info['tipo']}</div><div class="stat-lbl">Clasificación</div></div>
                <div class="stat-box"><div class="stat-val">{info['urgencia']}</div><div class="stat-lbl">Urgencia</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Ver descripción completa"):
            st.markdown(f"""
            <div class="info-row"><span class="info-lbl">Descripción</span><span class="info-val">{info['descripcion']}</span></div>
            <div class="info-row"><span class="info-lbl">Síntomas</span><span class="info-val">{info['sintomas']}</span></div>
            <div class="info-row"><span class="info-lbl">Causas</span><span class="info-val">{info['causas']}</span></div>
            <div class="info-row"><span class="info-lbl">Tratamiento</span><span class="info-val">{info['tratamiento']}</span></div>
            <div class="info-row"><span class="info-lbl">Recomendación</span><span class="info-val" style="color:{color}; font-weight:600;">{info['recomendacion']}</span></div>
            """, unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem; font-weight:600; color:#64748b; margin-top:1rem; text-transform:uppercase; letter-spacing:0.05em;">Distribución de probabilidades</div>', unsafe_allow_html=True)
        for cls, prob in zip(CLASES, r['probs']):
            nombre_c = INFO_CLASES[cls]['nombre'][:26]
            codigo_c = INFO_CLASES[cls]['codigo_display']
            pct = prob * 100
            fill_color = color if cls == r['clase'] else '#94a3b8'
            weight = '700' if cls == r['clase'] else '400'
            text_color = '#ffffff' if cls == r['clase'] else '#94a3b8'
            st.markdown(f"""
            <div style="margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                    <span style="font-weight:{weight}; color:{text_color};">{codigo_c} · {nombre_c}</span>
                    <span style="font-weight:{weight}; color:{fill_color};">{pct:.1f}%</span>
                </div>
                <div class="prob-bar"><div class="prob-fill" style="background:{fill_color}; width:{pct}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:3rem; text-align:center; color:#94a3b8;">
            <div style="font-size:2.5rem;">📊</div>
            <p style="margin:0.5rem 0 0; font-size:0.9rem; font-weight:500; color:#64748b;">Sin análisis aún</p>
            <p style="margin:0.3rem 0 0; font-size:0.8rem;">El diagnóstico aparecerá aquí</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
        <span style="font-size:0.72rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.05em;">💬 Asistente inteligente</span>
        <span class="badge badge-blue">NLP · Transformers</span>
    </div>
    """, unsafe_allow_html=True)
    chat_html = '<div class="chat-container">'
    if not st.session_state.chat_messages:
        chat_html += '<div style="text-align:center; color:#94a3b8; padding:3rem 1rem;"><div style="font-size:2rem;">🤖</div><p style="font-size:0.85rem; margin:0.5rem 0 0;">Analiza una imagen y luego<br>pregúntame lo que necesites.</p></div>'
    else:
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                chat_html += f'<div class="msg-user">{msg["content"]}</div>'
            else:
                content = msg["content"].replace('\n', '<br>')
                chat_html += f'<div class="msg-bot">{content}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
    if st.session_state.resultado:
        pregunta = st.chat_input("Escribe tu pregunta...")
        if pregunta:
            st.session_state.chat_messages.append({"role": "user", "content": pregunta})
            with st.spinner("Procesando con NLP..."):
                nlp = cargar_nlp()
                respuesta = consultar_agente(pregunta, st.session_state.resultado, nlp)
            st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})
            st.rerun()
    else:
        st.chat_input("Analiza una imagen primero...", disabled=True)
    st.markdown("""
    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px; padding:0.6rem 1rem; margin-top:0.8rem; font-size:0.75rem; color:#0369a1;">
        <b>Preguntas sugeridas:</b> "¿Es peligroso?", "¿Qué tratamiento hay?", "¿Cuándo ir al médico?", "¿Cómo proteger mi piel?"
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚕️ <b>Aviso médico:</b> DermAI es una herramienta académica (IRV2 + Soft Attention + NLP Semántico).
    <b>No constituye diagnóstico médico definitivo.</b> Consulta siempre a un dermatólogo certificado.
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Acerca de DermAI — Arquitectura y lesiones detectables"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Modelo de visión**\n\nInceptionResNetV2 + Soft Attention · Dataset HAM10000 (10,015 imágenes) · Exportado a ONNX")
    with c2:
        st.markdown("**Modelo de lenguaje**\n\nparaphrase-multilingual-MiniLM-L12-v2 · Sentence Transformers (HuggingFace) · Comprensión semántica multilingüe")
    st.markdown("---")
    cols = st.columns(2)
    for i, (cls, info) in enumerate(INFO_CLASES.items()):
        with cols[i % 2]:
            st.markdown(f"**{info['codigo_display']} · {info['nombre']}** — {info['tipo']}\n\n{info['descripcion']}\n\n---")

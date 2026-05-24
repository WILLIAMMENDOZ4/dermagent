from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
MODEL_PATH    = 'modelo_piel_IRV2_SA.h5'
DRIVE_FILE_ID = '1aPKzrRVGO49Z8-ZAlXLkF-bDsOslj85W'
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
# CAPAS PERSONALIZADAS
# ─────────────────────────────────────────────
class SoftAttention(Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x
        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.i_shape = input_shape
        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads)
        self.out_attention_maps_shape = input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]
        if not self.aggregate_channels:
            self.out_features_shape = input_shape[:-1] + (input_shape[-1] + (input_shape[-1] * self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape
        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d, initializer='he_uniform', name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,), initializer='zeros', name='bias_conv3d')
        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        exp_x = tf.expand_dims(x, axis=-1)
        c3d = K.conv3d(exp_x, kernel=self.kernel_conv3d, strides=(1, 1, self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d, self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)
        conv3d = tf.transpose(conv3d, perm=(0, 4, 1, 2, 3))
        conv3d = tf.squeeze(conv3d, axis=-1)
        conv3d = tf.reshape(conv3d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2]))
        softmax_alpha = tf.nn.softmax(conv3d, axis=-1)
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1], self.i_shape[2]))(softmax_alpha)
        if not self.aggregate_channels:
            exp_softmax_alpha = tf.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = tf.transpose(exp_softmax_alpha, perm=(0, 2, 3, 1, 4))
            x_exp = tf.expand_dims(x, axis=-2)
            u = kl.Multiply()([exp_softmax_alpha, x_exp])
            u = kl.Reshape(target_shape=(self.i_shape[1], self.i_shape[2], u.shape[-1] * u.shape[-2]))(u)
        else:
            exp_softmax_alpha = tf.transpose(softmax_alpha, perm=(0, 2, 3, 1))
            exp_softmax_alpha = tf.reduce_sum(exp_softmax_alpha, axis=-1)
            exp_softmax_alpha = tf.expand_dims(exp_softmax_alpha, axis=-1)
            u = kl.Multiply()([exp_softmax_alpha, x])
        o = kl.Concatenate(axis=-1)([u, x]) if self.concat_input_with_scaled else u
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]

    def get_config(self):
        config = super(SoftAttention, self).get_config()
        config.update({'ch': self.channels, 'm': self.multiheads, 'concat_with_x': self.concat_input_with_scaled, 'aggregate': self.aggregate_channels})
        return config


class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return [inp for inp in inputs]
        return inputs
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})
        return config


# ─────────────────────────────────────────────
# DESCARGAR Y CARGAR MODELO
# ─────────────────────────────────────────────
def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Descargando modelo desde Google Drive (puede tardar 1-2 min)..."):
            url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def cargar_modelo():
    descargar_modelo()
    return load_model(MODEL_PATH, custom_objects={
        'SoftAttention': SoftAttention,
        'CustomScaleLayer': CustomScaleLayer
    })


def preprocesar_imagen(img):
    img = img.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.inception_resnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predecir(model, img):
    x = preprocesar_imagen(img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASES[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────
# CHAT INTELIGENTE SIN API
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

    if any(w in p for w in ['peligro', 'grave', 'cancer', 'cáncer', 'maligno', 'malo', 'mortal', 'muero', 'mori']):
        if codigo == 'mel':
            return f"El **{nombre}** es la forma más peligrosa de cáncer de piel. Sin embargo, detectado a tiempo tiene muy buen pronóstico. Es fundamental que consultes a un dermatólogo lo antes posible.{disclaimer}"
        elif codigo in ['bcc', 'akiec']:
            return f"El **{nombre}** requiere atención médica, pero con tratamiento oportuno tiene excelente pronóstico. {rec}{disclaimer}"
        else:
            return f"El **{nombre}** es una lesión **benigna**, no es cancerosa ni peligrosa. Sin embargo, siempre es bueno hacer seguimiento. {rec}{disclaimer}"

    elif any(w in p for w in ['hacer', 'hago', 'debo', 'recomien', 'consejo', 'siguiente', 'pasos']):
        return f"Para el **{nombre}**, la recomendación es: {rec}\n\nAdemás te sugiero:\n- Fotografiar la lesión para detectar cambios\n- Evitar exposición solar directa sobre la zona\n- No manipular ni rascar la lesión{disclaimer}"

    elif any(w in p for w in ['síntoma', 'sintoma', 'duele', 'dolor', 'pica', 'ardor', 'molest', 'siento', 'aspecto']):
        return f"Los síntomas típicos del **{nombre}** incluyen: {sint}\n\nSi experimentas dolor intenso, sangrado o crecimiento rápido, consulta a un médico pronto.{disclaimer}"

    elif any(w in p for w in ['causa', 'origen', 'por qué', 'porque', 'produce', 'aparece', 'sale']):
        return f"El **{nombre}** generalmente aparece por: {causas}\n\nProtegerte del sol con bloqueador solar diario es clave para prevenir lesiones cutáneas.{disclaimer}"

    elif any(w in p for w in ['tratamiento', 'cura', 'eliminar', 'quitar', 'operar', 'medicina', 'crema']):
        return f"Para el **{nombre}**, las opciones de tratamiento incluyen: {trat}\n\nEl tratamiento específico lo determina el dermatólogo según las características de tu lesión.{disclaimer}"

    elif any(w in p for w in ['seguro', 'confianza', 'precis', 'exacto', 'correcto', 'probabilidad', 'modelo']):
        nivel = 'muy seguro' if confianza > 0.85 else 'moderadamente seguro' if confianza > 0.6 else 'poco seguro'
        return f"El modelo clasificó esta imagen como **{nombre}** con una confianza del **{confianza*100:.1f}%** — está {nivel} de su predicción.\n\nNingún modelo de IA reemplaza el diagnóstico de un dermatólogo, quien puede hacer exploración física y biopsia.{disclaimer}"

    elif any(w in p for w in ['urgent', 'rápido', 'médico', 'doctor', 'dermatólogo', 'hospital', 'cuándo', 'cuando']):
        if codigo in ['mel', 'bcc', 'akiec']:
            return f"Con un diagnóstico de **{nombre}**, la urgencia es **{urgencia}**. Busca cita con un dermatólogo lo antes posible, idealmente esta semana.{disclaimer}"
        else:
            return f"El **{nombre}** tiene urgencia **{urgencia}**. No es emergencia, pero es buena idea confirmar el diagnóstico con un dermatólogo.{disclaimer}"

    elif any(w in p for w in ['prevenir', 'prevención', 'evitar', 'proteger', 'sol', 'bloqueador']):
        return f"Para prevenir lesiones cutáneas como el **{nombre}**:\n- Usar bloqueador solar SPF 50+ todos los días\n- Evitar el sol entre 10am y 4pm\n- Usar ropa protectora y sombrero\n- Revisar tu piel mensualmente con la regla ABCDE\n- Visitar al dermatólogo una vez al año{disclaimer}"

    elif any(w in p for w in ['contagia', 'contagioso', 'pega', 'transmite']):
        return f"No, el **{nombre}** no es contagioso. Las lesiones cutáneas de este tipo no se transmiten de persona a persona.{disclaimer}"

    elif any(w in p for w in ['qué es', 'que es', 'explica', 'descripción', 'informa', 'cuéntame', 'dime']):
        return f"El **{nombre}** es: {desc}\n\nDetectado con una confianza del {confianza*100:.1f}%. {rec}{disclaimer}"

    else:
        return f"Sobre el **{nombre}** detectado:\n\n📋 {desc}\n\n🔬 Síntomas típicos: {sint}\n\n✅ Recomendación: {rec}{disclaimer}"


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

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
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
    archivo = st.file_uploader("Sube una imagen de la lesión", type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
    if archivo:
        img = Image.open(archivo)
        st.image(img, width=280)
        if st.button("🔍 Analizar"):
            with st.spinner("Analizando..."):
                try:
                    modelo = cargar_modelo()
                    clase, confianza, probs = predecir(modelo, img)
                    info = INFO_CLASES[clase]
                    st.session_state.resultado = {
                        'clase': clase, 'confianza': confianza, 'probs': probs, 'info': info,
                        'codigo': clase, 'nombre': info['nombre'], 'descripcion': info['descripcion'],
                        'urgencia': info['urgencia'], 'sintomas': info['sintomas'],
                        'causas': info['causas'], 'tratamiento': info['tratamiento'],
                        'recomendacion': info['recomendacion']
                    }
                    st.session_state.chat_history = []
                    st.session_state.chat_messages = []
                    msg_inicial = f"He analizado la imagen. El modelo detectó **{info['nombre']}** con una confianza del **{confianza*100:.1f}%**. ¿Tienes alguna pregunta sobre esta lesión?"
                    st.session_state.chat_messages.append({"role": "assistant", "content": msg_inicial})
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
            <p style="color:#9ca3af; margin:0; font-size:0.85rem;">Confianza: <strong style="color:white;">{r['confianza']*100:.1f}%</strong> &nbsp;·&nbsp; Urgencia: {info['urgencia']}</p>
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
            <p style="margin:0.5rem 0 0; font-size:0.9rem;">El resultado aparecerá aquí</p>
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
                <p style="font-size:0.85rem;">Analiza una imagen primero y luego<br>podrás hacerme preguntas sobre la lesión.</p>
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
    ⚠️ <strong>Aviso:</strong> Este asistente es una herramienta académica basada en IA. <strong>No reemplaza el diagnóstico médico profesional.</strong> Ante cualquier lesión sospechosa, consulta siempre a un dermatólogo certificado.
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ ¿Qué lesiones puede detectar el modelo?"):
    cols = st.columns(2)
    for i, (cls, info) in enumerate(INFO_CLASES.items()):
        with cols[i % 2]:
            st.markdown(f"**{cls.upper()} — {info['nombre']}**\n\n{info['descripcion']}\n\nUrgencia: {info['urgencia']}\n\n---")

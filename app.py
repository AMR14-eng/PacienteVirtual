from flask import Flask, request, jsonify, render_template
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Cargar el modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")

# Función para lematización y eliminación de palabras extra
def preprocesar_texto(texto):
    doc = nlp(texto.strip().lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# Frases por categoría para entrenar el modelo
frases_consulta = [
    "Hola, buen día, digame en que le puedo ayudar",
    "Cual es el motivo de su consulta",
    "En que puedo ayudarte hoy",
    "¡Hola! ¿En qué puedo asistirte hoy?",
    "¿Qué problema le trae hoy aquí?",
    "¿Cómo puedo ayudarte en este momento?",
    "¿Cuál es el motivo de tu consulta?",
    "¿Qué te trae por aquí?",
    "¡Buenos días! ¿En qué puedo ser útil?",
    "¿Hay algo en lo que pueda ayudarte?",
    "¡Hola! ¿Qué necesitas saber o resolver hoy?"
]
etiquetas_consulta = ["consulta"] * len(frases_consulta)

frases_frecuencia = [
    "Con que frecuencia los padece",
    "Cual es la frecuencia de estos dolores",
    "Estos dolores de cabeza ocurren mas de 15 días al mes",
    "Con que frecuencia los presenta",
    "Suelen ocurrir menos de 15 días al mes",
    "¿Cada cuánto tiempo tienes estos dolores?",
    "¿Con qué frecuencia experimentas estos síntomas?",
    "¿Los dolores aparecen muy seguido?",
    "¿Estos dolores ocurren varias veces al mes?",
    "¿Son dolores frecuentes o esporádicos?",
    "¿Podrías decirme cuántas veces al mes ocurre?",
    "¿Es algo que pasa todos los días o solo de vez en cuando?",
    "¿Los dolores se presentan constantemente o de forma ocasional?"
]
etiquetas_frecuencia = ["frecuencia"] * len(frases_frecuencia)

frases_intensidad = [
    "Podrías describir la intensidad del dolor",
    "Cual suele ser la intensidad de estos dolores",
    "Que tipo de dolor tiene",
    "¿Cómo calificarías la intensidad del dolor en una escala del 1 al 10?",
    "¿El dolor es leve, moderado o intenso?",
    "¿Qué tan fuerte es el dolor que sientes?",
    "¿El dolor es soportable o muy severo?",
    "¿Cómo describirías el nivel de dolor que tienes?",
    "¿Es un dolor suave o es incapacitante?",
    "¿Es un dolor leve que puedes ignorar o es muy fuerte?",
    "¿Sientes el dolor como algo ligero o como algo que te impide hacer actividades?"
]
etiquetas_intensidad = ["intensidad"] * len(frases_intensidad)

frases_localizacion = [
    "Puedes señalarme dónde sientes el dolor",
    "En que parte presenta el dolor",
    "En donde se localiza el dolor",
    "En donde se encuentra el dolor",
    "En donde le duele",
    "Podria decirme en que parte de la cabeza siente el dolor",
    "¿El dolor está en un lugar específico de la cabeza?",
    "¿Puedes indicar exactamente dónde te duele?",
    "¿Sientes el dolor en una zona en particular?",
    "¿El dolor está en un solo lado de la cabeza o en ambos?",
    "¿En qué parte del cuerpo sientes el dolor?",
    "¿Podrías describir la ubicación del dolor?",
    "¿Está el dolor más en la frente, sienes o parte trasera de la cabeza?",
    "¿El dolor está en la parte superior, inferior o en algún lado específico?",
    "¿Sientes el dolor más hacia el lado derecho, izquierdo o en el centro?",
    "¿El dolor se concentra en una zona o se distribuye por toda la cabeza?"
]
etiquetas_localizacion = ["localizacion"] * len(frases_localizacion)

frases_agravamiento = [
    "El dolor aumenta o disminuye con algo en particular",
    "Estos dolores empeoran con actividad fisica",
    "Se agrava con algo en especifico",
    "¿El dolor empeora en ciertas situaciones?",
    "¿Hay algo que haga que el dolor sea más fuerte?",
    "¿Notas que el dolor aumenta con alguna actividad?",
    "¿Sientes que ciertos movimientos o acciones agravan el dolor?",
    "¿Hay algo específico que haga que el dolor se intensifique?",
    "¿El dolor empeora con el estrés o el esfuerzo físico?",
    "¿Los síntomas aumentan con el ruido, la luz o algo más?",
    "¿Notas que el dolor es más fuerte en algún momento del día?",
    "¿El dolor empeora al realizar algún tipo de esfuerzo?",
    "¿Se agrava el dolor con el clima o algún alimento?"
]
etiquetas_agravamiento = ["agravamiento"] * len(frases_agravamiento)

frases_inicio = [
    "Cuando inició el dolor",
    "Desde cuando los presenta",
    "¿Desde hace cuánto tiempo comenzó el dolor?",
    "¿Recuerda cuándo empezó el dolor?",
    "¿Cuánto tiempo lleva sintiendo el dolor?",
    "¿Desde cuándo siente estas molestias?",
    "¿El dolor es reciente o lleva tiempo con él?",
    "¿Cuándo fue la primera vez que notó el dolor?",
    "¿Desde hace cuánto lo siente?"
]
etiquetas_inicio = ["inicio"] * len(frases_inicio)

frases_irradiacion = [
    "El dolor se irradia a algun otro punto o se mantiene fijo",
    "El dolor se le va a algun otro lugar o es fijo",
    "¿El dolor se mueve hacia algún lado?",
    "¿Siente que el dolor se expande a otra parte del cuerpo?",
    "¿El dolor se queda en el mismo lugar o cambia de ubicación?",
    "¿Se traslada el dolor a otras áreas?",
    "¿El dolor se mantiene fijo o lo siente en otras zonas?",
    "¿Siente el dolor en más de un lugar?",
    "¿El dolor se irradia o es localizado?"
]
etiquetas_irradiacion = ["irradiacion"] * len(frases_irradiacion)

frases_presion = [
    "Como es el dolor",
    "Podria describirme como es el dolor",
    "Como le duele",
    "Siente punzadas o es opresivo",
    "¿El dolor es punzante, opresivo o de otro tipo?",
    "¿Cómo describiría la sensación del dolor?",
    "¿Siente que el dolor es constante o intermitente?",
    "¿El dolor es como una presión, un ardor o algo diferente?",
    "¿El dolor es leve, moderado o intenso?",
    "¿Siente un dolor agudo o es más bien una molestia?",
    "¿Es un dolor que aprieta o como si pulsara?",
    "¿El dolor se siente como un pinchazo o más como una presión?"
]
etiquetas_presion = ["presion"] * len(frases_presion)

frases_sintomas = [
    "Aparte del dolor de cabeza presenta algun otro sintoma",
    "¿Tiene algún otro síntoma además del dolor?",
    "¿Presenta mareos, náuseas u otros síntomas?",
    "¿Ha notado algo más aparte del dolor de cabeza?",
    "¿Acompaña este dolor con otros malestares?",
    "¿Hay otros síntomas que se presenten junto al dolor?",
    "¿Tiene alguna molestia adicional al dolor de cabeza?",
    "¿Además del dolor, siente algo más fuera de lo común?"
]
etiquetas_sintomas = ["sintomas"] * len(frases_sintomas)

frases_medicamento = [
    "Has tomado algun medicamento",
    "Tomo ya algun medicamento",
    "¿Ha tomado algo para aliviar el dolor?",
    "¿Ha consumido algún analgésico o medicamento?",
    "¿Qué medicamentos ha usado para tratar el dolor?",
    "¿Ya tomó algo para este dolor?",
    "¿Está tomando algún tratamiento para el dolor?",
    "¿Ha recurrido a algún medicamento por su cuenta?",
    "¿Tomó alguna pastilla o medicamento recientemente?"
]
etiquetas_medicamento = ["medicamento"] * len(frases_medicamento)

frases_diagnostico = [
    "Podríamos estar hablando de una migraña episodica",
    "Su diagnostico es migraña episodica",
    "Usted tiene migraña episodica",
    "Usted tiene migraña episódica"
]
etiquetas_diagnostico = ["diagnostico"] * len(frases_diagnostico)

# Concatenar todas las frases y etiquetas
frases = (
    frases_consulta + frases_frecuencia + frases_intensidad + frases_localizacion +
    frases_agravamiento + frases_inicio + frases_irradiacion + frases_presion +
    frases_sintomas + frases_medicamento + frases_diagnostico
)
etiquetas = (
    etiquetas_consulta + etiquetas_frecuencia + etiquetas_intensidad +
    etiquetas_localizacion + etiquetas_agravamiento + etiquetas_inicio +
    etiquetas_irradiacion + etiquetas_presion + etiquetas_sintomas +
    etiquetas_medicamento + etiquetas_diagnostico
)

# Lematizar frases
def lematizar(frases, etiquetas):
    frases.extend([preprocesar_texto(frase) for frase in frases])
    etiquetas.extend(etiquetas)

lematizar(frases, etiquetas)

# Entrenar el modelo
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(frases, etiquetas)

# Diccionario de respuestas del paciente
respuestas = {
    "diagnostico": "Muchas gracias, doctor. Felicidades, tu diagnóstico es correcto!",
    "consulta": "He estado teniendo dolores de cabeza últimamente.",
    "intensidad": "Son bastante intensos.",
    "sintomas": "Sensibilidad a la luz y he tenido mareos y ansiedad.",
    "agravamiento": "Definitivamente empeoran con cualquier actividad fisica.",
    "frecuencia": "Suelen ocurrir menos de 15 días al mes.",
    "medicamento": "Sí, he tomado ibuprofeno un par de veces, pero solo me alivia temporalmente.",
    "localizacion": "El dolor se localiza en la parte frontal de la cabeza.",
    "inicio": "Inicio desde hace como 6 meses.",
    "irradiacion": "No se irradia.",
    "presion": "Siento como punzadas."
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

# Route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    texto_procesado = preprocesar_texto(text) 
    prediction = model.predict([texto_procesado])[0]
    response = respuestas.get(prediction, "¿Podría repetir la pregunta?")
    return jsonify({'prediction': prediction, 'response': response})

if __name__ == "__main__":
    app.run(debug=True)
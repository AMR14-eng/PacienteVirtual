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

# Datos iniciales de entrenamiento
frases = [
    preprocesar_texto("Cual es el motivo de su consulta"),
    preprocesar_texto("En que puedo ayudarte hoy"),
    preprocesar_texto("Cual es la frecuencia de estos dolores"),
    preprocesar_texto("Estos dolores de cabeza ocurren mas de 15 días al mes"),
    preprocesar_texto("Cual suele ser la intensidad de estos dolores"),
    preprocesar_texto("Estos dolores empeoran con actividad fisica"),
    preprocesar_texto("Has tomado algun medicamento"),
    preprocesar_texto("Podríamos estar hablando de una migraña episodica")
]
etiquetas = [
    "consulta", 
    "consulta", 
    "frecuencia", 
    "frecuencia", 
    "intensidad", 
    "agravamiento", 
    "medicamento", 
    "diagnostico"
]

# Crear un pipeline que haga el procesamiento de texto y el entrenamiento
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Entrenar el modelo por primera vez
model.fit(frases, etiquetas)

# Función para agregar nuevas frases y reentrenar el modelo
def agregar_y_reentrenar(nuevas_frases, nuevas_etiquetas):
    frases.extend([preprocesar_texto(frase) for frase in nuevas_frases])
    etiquetas.extend(nuevas_etiquetas)
    
    # Reentrenar el modelo desde cero
    model.fit(frases, etiquetas)
    print("Modelo reentrenado con los nuevos datos.")

# Frases por categoría
frases_consulta = [
    "Hola, buen día, digame en que le puedo ayudar",
    "Cual es el motivo de su consulta",
    "En que puedo ayudarte hoy"
]
etiquetas_consulta = ["consulta"] * len(frases_consulta)

frases_frecuencia = [
    "Con que frecuencia los padece",
    "Estos dolores de cabeza ocurren mas de 15 días al mes",
    "Con que frecuencia los presenta",
    "Suelen ocurrir menos de 15 días al mes"
]
etiquetas_frecuencia = ["frecuencia"] * len(frases_frecuencia)

frases_intensidad = [
    "Podrías describir la intensidad del dolor",
    "Cual suele ser la intensidad de estos dolores",
    "Que tipo de dolor tiene"
]
etiquetas_intensidad = ["intensidad"] * len(frases_intensidad)

frases_localizacion = [
    "Puedes señalarme dónde sientes el dolor",
    "En que parte presenta el dolor",
    "En donde se localiza el dolor",
    "En donde se encuentra el dolor",
    "En donde le duele",
    "Podria decirme en que parte de la cabeza siente el dolor"
]
etiquetas_localizacion = ["localizacion"] * len(frases_localizacion)

frases_agravamiento = [
    "El dolor aumenta o disminuye con algo en particular",
    "Estos dolores empeoran con actividad fisica",
    "Se agrava con algo en especifico"
]
etiquetas_agravamiento = ["agravamiento"] * len(frases_agravamiento)

frases_inicio = [
    "Cuando inició el dolor",
    "Desde cuando los presenta"
]
etiquetas_inicio = ["inicio"] * len(frases_inicio)

frases_irradiacion = [
    "El dolor se irradia a algun otro punto o se mantiene fijo",
    "El dolor se le va a algun otro lugar o es fijo"
]
etiquetas_irradiacion = ["irradiacion"] * len(frases_irradiacion)

frases_presion = [
    "Como es el dolor",
    "Podria describirme como es el dolor",
    "Como le duele",
    "Siente punzadas o es opresivo"
]
etiquetas_presion = ["presion"] * len(frases_presion)

frases_sintomas = [
    "Aparte del dolor de cabeza presenta algun otro sintoma"
]
etiquetas_sintomas = ["sintomas"] * len(frases_sintomas)

frases_medicamento = [
    "Has tomado algun medicamento",
    "Tomo ya algun medicamento"
]
etiquetas_medicamento = ["medicamento"] * len(frases_medicamento)

frases_diagnostico = [
    "Podríamos estar hablando de una migraña episodica",
    "Su diagnostico es migraña episodica",
    "Usted tiene migraña episodica"
]
etiquetas_diagnostico = ["diagnostico"] * len(frases_diagnostico)

# Concatenar todas las frases y etiquetas
frases_nuevas = (
    frases_consulta + frases_frecuencia + frases_intensidad + frases_localizacion +
    frases_agravamiento + frases_inicio + frases_irradiacion + frases_presion +
    frases_sintomas + frases_medicamento + frases_diagnostico
)
etiquetas_nuevas = (
    etiquetas_consulta + etiquetas_frecuencia + etiquetas_intensidad +
    etiquetas_localizacion + etiquetas_agravamiento + etiquetas_inicio +
    etiquetas_irradiacion + etiquetas_presion + etiquetas_sintomas +
    etiquetas_medicamento + etiquetas_diagnostico
)

# Llamar a la función para agregar las nuevas frases y etiquetas, y luego reentrenar
agregar_y_reentrenar(frases_nuevas, etiquetas_nuevas)

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

# Función para predecir la intención de una frase
def predecir_intencion(texto):
    texto_procesado = preprocesar_texto(texto)
    prediccion = model.predict([texto_procesado])[0]
    return prediccion

# Función para responder según la predicción
def respuesta_paciente(prediccion):
    respuesta = respuestas.get(prediccion, "¿Podría repetir la pregunta?")
    print("Paciente:", respuesta)
    # Si el diagnóstico es correcto se detiene el flujo
    return prediccion != "diagnostico"

# Main
print("Paciente: Buenos días, doctor.")
while True:
    input_doctor = input("Doctor: ")
    prediccion = predecir_intencion(input_doctor)
    if not respuesta_paciente(prediccion):
        break  # Detener el bucle si se ha diagnosticado correctamente

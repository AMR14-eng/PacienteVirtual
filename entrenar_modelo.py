from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Preprocesamiento básico
def preprocesar_texto(texto):
    return texto.lower().strip()

# Frases por categoría (diversas y balanceadas)
frases_por_categoria = {
    "consulta": [
        "¿En qué puedo ayudarte hoy?",
        "¿Cuál es el motivo de tu consulta?",
        "¿Qué te trae por aquí?",
        "¿Cómo puedo asistirte?",
        "¿Qué problema presentas hoy?",
        "Hola, ¿qué deseas consultar?",
        "¿Qué inquietud de salud tienes?",
        "¿Cómo puedo ayudarte con tu dolor?",
        "¿Vienes por algún malestar específico?",
        "¿Deseas hablar sobre algún síntoma?"
    ],
    "frecuencia": [
        "¿Con qué frecuencia sientes el dolor?",
        "¿Ocurre seguido o es esporádico?",
        "¿Cada cuántos días tienes los síntomas?",
        "¿Esto pasa seguido?",
        "¿Sucede más de una vez por semana?",
        "¿El dolor es recurrente?",
        "¿Te pasa todos los días o solo algunos?",
        "¿Con qué regularidad ocurre esto?",
        "¿Es algo frecuente?",
        "¿Pasa de forma repetitiva?"
    ],
    "intensidad": [
        "¿Qué tan fuerte es el dolor?",
        "¿Cómo calificarías la intensidad?",
        "¿Es un dolor leve o severo?",
        "¿En una escala del 1 al 10, cuánto duele?",
        "¿Es incapacitante?",
        "¿Puedes soportarlo o te impide hacer cosas?",
        "¿Te afecta mucho el dolor?",
        "¿Es algo leve o realmente fuerte?",
        "¿Se siente suave o agudo?",
        "¿El dolor es moderado?"
    ],
    "localizacion": [
        "¿Dónde sientes el dolor exactamente?",
        "¿El dolor está en la cabeza o en otra parte?",
        "¿Puedes señalarme la zona?",
        "¿Se localiza en algún punto específico?",
        "¿En qué parte del cuerpo?",
        "¿Es en un solo lado o ambos?",
        "¿Está en la frente, sien o nuca?",
        "¿Dolor en el cuello también?",
        "¿Es frontal, lateral o posterior?",
        "¿El dolor se concentra en una región?"
    ],
    "agravamiento": [
        "¿Hay algo que empeora el dolor?",
        "¿Se intensifica al moverte?",
        "¿La luz o el ruido lo agravan?",
        "¿Se agrava con el estrés?",
        "¿Comer afecta el dolor?",
        "¿Qué factores lo empeoran?",
        "¿Alguna actividad lo hace peor?",
        "¿Se incrementa con el clima?",
        "¿El ejercicio lo agrava?",
        "¿Algo lo hace más difícil de soportar?"
    ],
    "inicio": [
        "¿Desde cuándo tienes este dolor?",
        "¿Cuándo comenzaron los síntomas?",
        "¿Es reciente o lleva tiempo?",
        "¿Cuándo notaste el primer síntoma?",
        "¿Desde hace cuántos días empezó?",
        "¿Llevas mucho con esto?",
        "¿El malestar empezó esta semana?",
        "¿Es algo nuevo o crónico?",
        "¿El dolor es de aparición reciente?",
        "¿Puedes recordar cuándo inició?"
    ],
    "irradiacion": [
        "¿El dolor se expande a otras zonas?",
        "¿Se mueve a otras partes del cuerpo?",
        "¿Es fijo o cambia de lugar?",
        "¿Se irradia hacia el cuello u hombros?",
        "¿El dolor viaja a otras áreas?",
        "¿Lo sientes en más de un lugar?",
        "¿Se queda en un solo sitio?",
        "¿El dolor se desplaza?",
        "¿Va hacia la espalda?",
        "¿El malestar se mueve o es localizado?"
    ],
    "descripcion_dolor": [
        "¿Cómo describirías el dolor?",
        "¿Es punzante, opresivo o quemante?",
        "¿Sientes presión o ardor?",
        "¿Es constante o intermitente?",
        "¿Es agudo o leve?",
        "¿Te arde o pulsa?",
        "¿Se siente como una presión o golpe?",
        "¿Cómo es la sensación?",
        "¿Es dolor punzante o latente?",
        "¿El dolor parece eléctrico o difuso?"
    ],
    "sintomas": [
        "¿Tienes otros síntomas además del dolor?",
        "¿Hay náuseas, vómitos o mareos?",
        "¿Acompañado de fiebre o visión borrosa?",
        "¿Algo más te molesta?",
        "¿Presentas otros signos clínicos?",
        "¿Tienes molestias adicionales?",
        "¿Síntomas como debilidad o confusión?",
        "¿Aparecen otros malestares?",
        "¿Experimentas otros cambios físicos?",
        "¿Además del dolor, qué más sientes?"
    ],
    "medicamento": [
        "¿Has tomado algo para el dolor?",
        "¿Usaste algún analgésico?",
        "¿Qué medicamento tomaste?",
        "¿Has recurrido a pastillas?",
        "¿Tomaste algo por tu cuenta?",
        "¿Estás bajo tratamiento médico?",
        "¿Te automedicaste?",
        "¿Has usado ibuprofeno o paracetamol?",
        "¿Tomaste algo recientemente?",
        "¿Ya habías tratado este dolor?"
    ],
    "diagnostico": [
        "Podría tratarse de migraña",
        "Es posible que sea cefalea tensional",
        "El diagnóstico puede ser neuralgia",
        "Todo indica una migraña común",
        "Podría ser jaqueca con aura",
        "Esto parece una cefalea crónica",
        "Se sospecha de migraña episódica",
        "Podemos concluir que es jaqueca",
        "Podría tratarse de una cefalea mixta",
        "Mi diagnóstico sería migraña"
    ]
}

# Preparar datos
frases = []
etiquetas = []
for categoria, ejemplos in frases_por_categoria.items():
    frases.extend([preprocesar_texto(f) for f in ejemplos])
    etiquetas.extend([categoria] * len(ejemplos))

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(frases, etiquetas, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
modelo.fit(X_train, y_train)

# Evaluar modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(modelo, "modelo_entrenado.pkl")
print("Modelo entrenado y guardado con éxito.")

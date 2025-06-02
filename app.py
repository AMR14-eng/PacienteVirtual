from flask import Flask, request, jsonify, render_template
import joblib
import psycopg2
import os
from dotenv import load_dotenv
from openai import OpenAI 
from preprocesamiento import preprocesar_texto

load_dotenv()

app = Flask(__name__)
modelo = joblib.load("modelo_entrenado.pkl")

# Configura API OpenAI apuntando a Groq
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url is None:
        raise ValueError("DATABASE_URL no está configurada")
    return psycopg2.connect(db_url)

# Construye el prompt a partir de los síntomas del paciente
def construir_prompt(fila, intencion):
    prompt = (
        "Eres un paciente en una consulta médica. Responde como un humano real, con emociones, y de forma coherente.\n"
        "pero responde específicamente a la pregunta que te hace el doctor según el motivo de la consulta.\n\n"
        "Tus síntomas actuales y antecedentes médicos son:\n\n"
    )
    for columna, valor in fila.items():
        if columna != "id":  # Ignorar ID
            prompt += f"{columna.capitalize()}: {valor}\n"

    prompt += f"\nEl doctor te pregunta sobre: {intencion}\n"
    prompt += "Responde con información relevante y concreta, sin añadir cosas fuera del tema.\n\n"
    prompt += "Doctor: {pregunta}\nPaciente:"
    return prompt

# Pregunta al modelo LLaMA 3
def preguntar_a_llama(prompt_con_pregunta):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": (
                "Actúa como un paciente real en una consulta médica. "
                "Responde de forma natural, clara y concisa. "
                "No incluyas descripciones de acciones entre paréntesis. "
                "No rolees comportamientos físicos. "
                "Solo responde con tus síntomas y cómo te sientes, como si hablaras normalmente con un doctor."
                "Limita tu respuesta a un máximo de 3 oraciones."
            )},
            {"role": "user", "content": prompt_con_pregunta}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        texto = data.get("text", "")

        if not texto:
            return jsonify({"error": "Texto vacío"}), 400

        procesado = preprocesar_texto(texto)
        intencion = modelo.predict([procesado])[0]

        conn = get_connection()
        cur = conn.cursor()

        # Obtener paciente aleatorio
        cur.execute("SELECT * FROM paciente ORDER BY RANDOM() LIMIT 1")
        colnames = [desc[0] for desc in cur.description]
        values = cur.fetchone()
        conn.close()

        if not values:
            return jsonify({"error": "No se encontró paciente."}), 404

        paciente_info = dict(zip(colnames, values))

        # Si la intención es diagnóstico, comparar texto con el diagnóstico real
        if intencion == "diagnostico":
            diagnostico_real = paciente_info.get("diagnostico", "").strip().lower()
            diagnostico_usuario = texto.strip().lower()

            # Intenta extraer el diagnóstico desde la frase
            if diagnostico_real in diagnostico_usuario:
                es_correcto = True
            else:
                es_correcto = False

            return jsonify({
                "intencion": intencion,
                "respuesta": texto,
                "end": es_correcto,
                "respuesta_correcta": diagnostico_real
            })

        # Si no es diagnóstico, simular respuesta del paciente
        prompt_base = construir_prompt(paciente_info, intencion)
        prompt_con_pregunta = prompt_base.replace("{pregunta}", texto)
        respuesta = preguntar_a_llama(prompt_con_pregunta)

        return jsonify({
            "intencion": intencion,
            "respuesta": respuesta,
            "diagnostico": paciente_info.get("diagnostico", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
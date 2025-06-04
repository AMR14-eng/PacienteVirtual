from flask import Flask, request, jsonify, render_template, session
import joblib
import psycopg2
import os
from dotenv import load_dotenv
from openai import OpenAI 
from preprocesamiento import preprocesar_texto

load_dotenv()

app = Flask(__name__)
modelo = joblib.load("modelo_entrenado.pkl")
app.secret_key = os.getenv("SECRET_KEY")

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
        "Eres un paciente en una consulta médica. Responde solo con tus síntomas y cómo te sientes actualmente.\n"
        "Tus síntomas actuales son:\n"
    )
    for columna, valor in fila.items():
        if columna not in ("id", "diagnostico"):  # Excluye id y diagnóstico
            prompt += f"{columna.capitalize()}: {valor}\n"

    prompt += f"\nEl doctor te pregunta sobre: {intencion}\n"
    prompt += "Doctor: {pregunta}\nPaciente:"
    return prompt

# Recibe prompt y lo envia al modelo LLaMA 3
def preguntar_a_llama(prompt_con_pregunta):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": (
                "Eres un paciente real en una consulta médica. "
                "Responde solo describiendo tus síntomas actuales y cómo te sientes. "
                "No digas nombres de enfermedades ni diagnósticos. "
                "No incluyas descripciones de acciones ni explicaciones médicas. "
                "Limita la respuesta a máximo 50 palabras, con un tono natural y claro."
                "No menciones diagnósticos previos, nombres de enfermedades ni antecedentes médicos."
                "No añadas explicaciones ni información fuera del tema."
                "Sé breve y claro, máximo 2-3 oraciones."
            )},
            {"role": "user", "content": prompt_con_pregunta}
        ],
        temperature=0.3,
        max_tokens=100
    )
    return response.choices[0].message.content


@app.route("/")
def home():
    session['vidas'] = 3
    session['puntos'] = 0
    session['pacientes_usados'] = []
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

        if "paciente" not in session:
            usados = tuple(session.get('pacientes_usados', [])) or (0,)
            cur.execute("SELECT * FROM paciente WHERE id NOT IN %s ORDER BY RANDOM() LIMIT 1", (usados,))
            values = cur.fetchone()
            if not values:
                conn.close()
                return jsonify({"error": "No se encontró paciente."}), 404
            colnames = [desc[0] for desc in cur.description]
            paciente_info = dict(zip(colnames, values))
            session["paciente"] = paciente_info
        else:
            paciente_info = session["paciente"]

        conn.close()


        # Si la intención es diagnóstico, comparar texto con el diagnóstico real
        if intencion == "diagnostico":
            diagnostico_real = paciente_info.get("diagnostico", "").strip().lower()
            diagnostico_usuario = texto.strip().lower()

            es_correcto = diagnostico_real in diagnostico_usuario

            if es_correcto:
                session['puntos'] += 10
                session['pacientes_usados'].append(paciente_info["id"])

                # Buscar nuevo paciente que no haya sido usado
                conn = get_connection()
                cur = conn.cursor()
                usados = tuple(session['pacientes_usados']) or (0,)  # evitar error de tupla vacía
                cur.execute(
                    "SELECT * FROM paciente WHERE id NOT IN %s ORDER BY RANDOM() LIMIT 1", (usados,)
                )
                colnames = [desc[0] for desc in cur.description]
                values = cur.fetchone()
                conn.close()

                if not values:
                    session.pop("paciente", None)
                    return jsonify({
                        "intencion": intencion,
                        "respuesta": "¡Felicidades! Has diagnosticado a todos los pacientes disponibles.",
                        "end": True,
                        "puntos": session['puntos'],
                        "vidas": session['vidas'],
                        "completo": True
                    })

                nuevo_paciente = dict(zip(colnames, values))
                session["paciente"] = nuevo_paciente

                return jsonify({
                    "intencion": intencion,
                    "respuesta": "¡Bien hecho! Acertaste en el diagnóstico. Es hora de continuar con el siguiente paciente.",
                    "end": True,
                    "puntos": session['puntos'],
                    "vidas": session['vidas'],
                    "completo": False
                })

            else:
                session['vidas'] -= 1
                juego_terminado = session['vidas'] <= 0
                if juego_terminado:
                    session.clear()
                return jsonify({
                    "intencion": intencion,
                    "respuesta": "Ese no es el diagnóstico correcto. Intenta de nuevo.",
                    "end": False,
                    "vidas": session.get('vidas', 0),
                    "puntos": session.get('puntos', 0),
                    "game_over": juego_terminado
                })


        # Si no es diagnóstico, simular respuesta del paciente
        prompt_base = construir_prompt(paciente_info, intencion)
        prompt_con_pregunta = prompt_base.replace("{pregunta}", texto)
        respuesta = preguntar_a_llama(prompt_con_pregunta)

        return jsonify({
            "intencion": intencion,
            "respuesta": respuesta,
            "puntos": session['puntos'],
            "vidas": session['vidas'],
            "diagnostico": paciente_info.get("diagnostico", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
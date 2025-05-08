from flask import Flask, request, jsonify, render_template
import joblib
import psycopg2
from preprocesamiento import preprocesar_texto

app = Flask(__name__)
modelo = joblib.load("modelo_entrenado.pkl")

def get_connection():
    return psycopg2.connect(
        dbname="bot_medico",
        user="postgres",
        password="LLM28",  # Considera usar variables de entorno para mayor seguridad
        host="localhost",
        port="5432"
    )

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
        intencion = modelo.predict([procesado])[0]  # Ejemplo: 'frecuencia', 'intensidad', etc.

        conn = get_connection()
        cur = conn.cursor()

        # Elige enfermedad Aleatoria
        cur.execute("SELECT diagnostico FROM paciente ORDER BY RANDOM() LIMIT 1")
        resultado = cur.fetchone()
        # Establece la enfermedad
        if resultado:
            enfermedad = resultado[0]
            enfermedad_procesada = preprocesar_texto(enfermedad)
        else:
            return jsonify({"intencion": intencion, "respuesta": "No hay enfermedades registradas."})

        # Verifica que la intención sea una columna válida
        columnas_validas = [
            "consulta", "frecuencia", "intensidad", "sintomas", "localizacion",
            "agravamiento", "medicamentos", "inicio", "irradiacion",
            "descripcion_dolor", "diagnostico"
        ]

        if intencion not in columnas_validas:
            cur.close()
            conn.close()
            return jsonify({"intencion": intencion, "respuesta": "Lo siento, no puedo responder a eso aún."})
        
        # Consulta dinámica: obtiene el valor de la columna con nombre igual a la intención siempre que la intencion no sea diagnostico
        if intencion != "diagnostico":
            query = f"SELECT {intencion} FROM paciente WHERE diagnostico = %s LIMIT 1"
            cur.execute(query, (enfermedad,))
            resultado = cur.fetchone()
            cur.close()
            conn.close()

            if resultado:
                valor = resultado[0]
                return jsonify({"intencion": intencion, "respuesta": valor})
            else:
                return jsonify({"intencion": intencion, "respuesta": "No encontré información para esa enfermedad."})
        else:
            if enfermedad_procesada in procesado:
                return jsonify({"intencion": intencion, "end": True})
            else:
                return jsonify({"intencion": intencion, "end": False, "respuesta_correcta": enfermedad})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


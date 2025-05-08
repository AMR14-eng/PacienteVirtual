from flask import Flask, request, jsonify, render_template
import joblib
import psycopg2
import os
from dotenv import load_dotenv
from preprocesamiento import preprocesar_texto

# Cargar variables de entorno desde .env (funciona localmente)
load_dotenv()

app = Flask(_name_)
modelo = joblib.load("modelo_entrenado.pkl")

def get_connection():
    db_url = os.getenv("DATABASE_URL")
    print("Conectando a:", db_url)  # Solo para depuración (puedes borrar luego)
    if db_url is None:
        raise ValueError("DATABASE_URL no está configurada")
    return psycopg2.connect(db_url)

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

        cur.execute("SELECT diagnostico FROM paciente ORDER BY RANDOM() LIMIT 1")
        resultado = cur.fetchone()

        if resultado:
            enfermedad = resultado[0]
            enfermedad_procesada = preprocesar_texto(enfermedad)
        else:
            return jsonify({"intencion": intencion, "respuesta": "No hay enfermedades registradas."})

        columnas_validas = [
            "consulta", "frecuencia", "intensidad", "sintomas", "localizacion",
            "agravamiento", "medicamentos", "inicio", "irradiacion",
            "descripcion_dolor", "diagnostico"
        ]

        if intencion not in columnas_validas:
            cur.close()
            conn.close()
            return jsonify({"intencion": intencion, "respuesta": "Lo siento, no puedo responder a eso aún."})

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

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

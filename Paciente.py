import spacy

# Cargar el modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")

# Palabras clave que el bot reconoce como síntomas
palabras_clave_sintomas = ['siente', 'síntomas', 'consulta', 'encuentras']
# Palabras clave que el bot reconoce como descripción de síntomas
palabras_clave_detalles = ['describe', 'parte', 'describirias', 'desccribe', 'area']
# Palabras clave que el bot reconoce como nauseas o mareos
palabras_clave_nauseas = ['náuseas', 'nauseas', 'mareos']
# Palabras clave que el bot reconoce como dias de dolor
palabras_clave_tiempo = ['tiempo', 'dias']
# Palabras clave que el bot reconoce como agravamiento del dolor
palabras_clave_agravamiento = ['empeora', 'notado', 'estrés', 'estres', 'trabajando', 'estudiando', 'presión']
# Palabras clave que el bot reconoce como medicamento tomado
palabras_clave_medicamento = ['tomando', 'medicamento', 'remedio']
# Palabras clave que el bot reconoce como diagnóstico
palabras_clave_diagnostico = ['cefalea','tensional']

# Función para detectar palabras clave en el input del doctor
def detectar_palabras_clave(texto):
    doc = nlp(texto.strip().lower())  # Convertir a minúsculas y eliminar espacios extra
    
    # Extraer tokens del input y comprobar si coinciden con las palabras clave
    sintoma_detectado = [token.text for token in doc if token.text in palabras_clave_sintomas]
    detalles_detectado = [token.text for token in doc if token.text in palabras_clave_detalles]
    nauseas_detectado = [token.text for token in doc if token.text in palabras_clave_nauseas]
    tiempo_detectado = [token.text for token in doc if token.text in palabras_clave_tiempo]
    agravamiento_detectado = [token.text for token in doc if token.text in palabras_clave_agravamiento]
    medicamento_detectado = [token.text for token in doc if token.text in palabras_clave_medicamento]
    diagnostico_detectado = [token.text for token in doc if token.text in palabras_clave_diagnostico]
    
    # Responder basado en prioridades: primero diagnóstico, luego medicamento, etc.
    if len(diagnostico_detectado) >= 2:  # Verificar que ambas palabras clave estén presentes
        print("Paciente: ¡Felicidades! Has diagnosticado correctamente a tu paciente.")
        return False  # Cortar el flujo si se detecta el diagnóstico
    elif medicamento_detectado:
        print("Paciente: Sí, he tomado ibuprofeno un par de veces, pero solo me alivia temporalmente.")
    elif agravamiento_detectado:
        print("Paciente: Me parece que empeora cuando estoy bajo presión en el trabajo o si paso muchas horas frente a la computadora.")
    elif detalles_detectado:
        print("Paciente: Es un dolor opresivo, lo siento más en la parte frontal y a los lados de la cabeza. No es muy intenso, pero es constante y molesto. No tengo náuseas ni mareos. \nCreo que llevo así unos cuatro o cinco días.")
    elif nauseas_detectado:
        print("Paciente: No tengo náuseas ni mareos.")
    elif tiempo_detectado:
        print("Paciente: Creo que llevo así unos cuatro o cinco días.")
    elif sintoma_detectado:
        print("Paciente: He venido porque llevo varios días con dolor de cabeza.")
    else:
        print("Paciente: ¿Me puede repetir lo que dijo?")
    
    return True  # Continuar el flujo si no se ha diagnosticado aún

# Main
print("Paciente: Hola, buenos días doctor.")
while True:
    input_doctor = input("Doctor: ")
    if not detectar_palabras_clave(input_doctor):
        break  # Detener el bucle si se ha diagnosticado al paciente

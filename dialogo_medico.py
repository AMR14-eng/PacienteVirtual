import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class DialogoMedico:
    def __init__(self):
        # Cargar el modelo de lenguaje en español
        self.nlp = spacy.load("es_core_news_sm")
        
        # Definir frases y etiquetas
        self.frases, self.etiquetas = self._crear_datos_entrenamiento()
        
        # Lematizar frases y etiquetas
        self.frases = [self._preprocesar_texto(frase) for frase in self.frases]
        
        # Entrenar el modelo
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.model.fit(self.frases, self.etiquetas)
        
        # Diccionario de respuestas del paciente
        self.respuestas = {
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

    def _preprocesar_texto(self, texto):
        doc = self.nlp(texto.strip().lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
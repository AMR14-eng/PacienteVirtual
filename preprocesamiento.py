import spacy
import unicodedata

nlp = spacy.load("es_core_news_sm")

def preprocesar_texto(texto):
    doc = nlp(texto.strip().lower())
    # Eliminar espacios y pasar a minúsculas
    texto = texto.strip().lower()
    
    # Procesar con spaCy
    doc = nlp(texto)
    
    # Lematizar, eliminar stopwords y quitar acentos todo en un solo paso
    tokens = []
    for token in doc:
        if not token.is_stop:
            lema = token.lemma_
            lema_sin_acentos = ''.join(
                c for c in unicodedata.normalize("NFD", lema)
                if unicodedata.category(c) != "Mn"
            )
            tokens.append(lema_sin_acentos)
    
    return " ".join(tokens)
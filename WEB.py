from flask import Flask

app=Flask(__name__)
#decorador para asociar la ruta con la función
@app.route("/")
def index():
    return "<h1>Hola a todos!!</h1>"

if __name__=='__main__':
    app.run()

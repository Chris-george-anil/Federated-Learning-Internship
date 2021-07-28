from flask import Flask
import server as s
import dp_server as dp
app = Flask(__name__)

@app.route("/")
def hello_world():
    dp
    return "<p>Hello, World!</p>"
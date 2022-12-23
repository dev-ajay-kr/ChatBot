from flask import Flask, request, Response, jsonify
from chat import response as chatBotResponse
from flask_cors import CORS

app = Flask("__name__")
CORS(app)

@app.route("/", methods=['GET'])
def checkIfApiIsWorkingFine():
    return Response("This is the response from your chatbot api")

@app.route("/predict", methods=['POST'])
def getAnswer():
    userInput = request.get_json()

    userMessage = userInput["message"]

    chatbotResp = chatBotResponse(userMessage)

    return jsonify(chatbotResp)

@app.errorhandler(Exception)
def basic_error_handlor(e):
    return jsonify("{Error: Some error has ocurred" + str(e) + "}")
    


if __name__=="__main__":
    app.run(debug=True)
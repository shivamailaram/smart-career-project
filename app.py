from flask import Flask, render_template, request
from model import train_model

app = Flask(__name__)
model, le = train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        math = int(request.form['math'])
        science = int(request.form['science'])
        english = int(request.form['english'])
        interest = request.form['interest'].lower()

        interest_encoded = le.transform([interest])[0]

        prediction = model.predict([[math, science, english, interest_encoded]])
        result = prediction[0]

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

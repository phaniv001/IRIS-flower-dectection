from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
lr = pickle.load(open('iris.pkl','rb'))
@app.route("/")
def index():
    return render_template('index_1.html')


@app.route("/predict",methods=["POST"])

def predict():
    if request.method == "POST":
        spl = request.form['spl']
        spw = request.form['spw']
        ptl = request.form['ptl']
        ptw = request.form['ptw']

        data = [[float(spl), float(spw), float(ptl), float(ptw)]]
        prediction = lr.predict(data)[0]
    return render_template("index_1.html", prediction = prediction)


if __name__ == '__main__':
    app.run(host = '0.0.0.0')
    #app.run(debug=True)
    



from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('temp.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        polyuria = request.form['polyuria']
        polydipsia = request.form['polydipsia']
        suddenloss = request.form['suddenloss']
        weakness = request.form['weakness']
        polyphagia = request.form['polyphagia']
        genital = request.form['genital']
        visual = request.form['visual']
        itching = request.form['genital']
        irritability = request.form['irritability']
        delayed = request.form['delayed']
        partial = request.form['partial']
        muscle = request.form['muscle']
        alopecia = request.form['alopecia']
        obesity = request.form['obesity']
        
        data = np.array([[age,gender,polyuria,polydipsia,suddenloss,weakness,polyphagia,genital,visual,itching,irritability,delayed,partial,muscle,alopecia,obesity]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
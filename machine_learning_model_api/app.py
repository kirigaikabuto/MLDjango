import pickle

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
filename = './ml_model/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
full_data = [
    {
        "name": "Iris-setosa",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    },
    {
        "name": "Iris-versicolor",
        "image": "https://lincspplants.co.uk/wp-content/uploads/2017/02/product_i_r_iris_versicolor_3.jpg"
    },
    {
        "name": "Iris-virginica",
        "image": "https://s3.amazonaws.com/eit-planttoolbox-prod/media/images/Iris-virginica--Justin-Meissen--CC-BY-SA.jpg",
    },
]


@app.route('/api/flowers/identify/', methods=["POST"])
def identify():
    data = request.get_json()
    sepal_length = float(data['sepal_length'])
    sepal_width = float(data['sepal_width'])
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])
    predicted_flower_name = loaded_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    response = {}
    print(predicted_flower_name)
    for el in full_data:
        if el["name"] == predicted_flower_name[0]:
            response = el
            break
    return jsonify(response)


@app.route("/api/flowers/", methods=["GET"])
def get_flowers():
    return jsonify(full_data)


@app.route("/", methods=["GET"])
def main_page():
    return render_template("index.html", flowers=full_data)


if __name__ == "__main__":
    app.run(debug=True)

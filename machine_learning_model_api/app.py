import pickle

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
filename = '../finalized_model.sav'
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
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1200px-Iris_virginica_2.jpg",
    },
]


@app.route('/api/flowers/identify', methods=["POST"])
def identify():
    data = request.get_json()
    sepal_length = float(data['sepal_length'])
    sepal_width = float(data['sepal_width'])
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])
    predicted_flower_name = loaded_model.predict[sepal_length, sepal_width, petal_length, petal_width]
    print(predicted_flower_name)
    return jsonify(data)


@app.route("/api/flowers/", methods=["GET"])
def get_flowers():
    return jsonify(full_data)


@app.route("/", methods=["GET"])
def main_page():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

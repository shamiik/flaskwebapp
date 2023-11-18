from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0: 'Ripe', 1: 'UnRipe'}

model = load_model('model_resnet50.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)/255.0
    # i = i.reshape(1, 100, 100, 3)
    p = model.predict_classes(i)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Stay with me. Shamik"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


# if __name__ == "__main__":
#     app.run()

if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)

# if __name__ == "__main__":
#     app.run(debug=False, host='0.0.0.0')

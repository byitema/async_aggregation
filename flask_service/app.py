from flask import Flask, request
from image_processing import process_image

app = Flask(__name__)


@app.route('/process_image', methods=['POST'])
def process_img():
    image = request.files['image']
    model = int(request.form.get('model'))

    result = process_image(image, model)

    return result


if __name__ == '__main__':
    app.run()

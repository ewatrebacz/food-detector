import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import os

app = Flask(__name__)

food_list = ['ice_cream', 'samosa', 'tuna_tartare', 'seaweed_salad', 'shrimp_and_grits', 'steak',
             'red_velvet_cake', 'waffles', 'gyoza', 'lobster_roll_sandwich', 'huevos_rancheros',
             'spaghetti_bolognese', 'poutine', 'ravioli', 'lobster_bisque', 'risotto',
             'strawberry_shortcake', 'hot_and_sour_soup', 'spring_rolls', 'sashimi', 'paella',
             'miso_soup', 'hot_dog', 'pulled_pork_sandwich', 'panna_cotta', 'pad_thai', 'tiramisu',
             'takoyaki', 'macarons', 'apple_pie', 'scallops', 'mussels', 'spaghetti_carbonara',
             'omelette', 'sushi', 'hummus', 'pork_chop', 'tacos', 'hamburger', 'pancakes',
             'prime_rib', 'pizza', 'nachos', 'macaroni_and_cheese', 'ramen', 'lasagna',
             'peking_duck', 'pho', 'oysters', 'onion_rings']
food_list = sorted(food_list)

#K.clear_session()
#model_best = load_model('model.keras', compile=False)
print(os.system("pwd"))
print(os.system("ls -l"))

def predict_class(model, img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    return pred_value

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400

    img_data = request.json['image']
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((299, 299))
     
    print(os.system("pwd"))
    print(os.system("ls -l"))
    #prediction = predict_class(model_best, img)
    #print(prediction)
    #plt.imshow(img)
    #plt.axis('off')
    #plt.title(prediction)

    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    #plt.close()

    return jsonify({"prediction": img_data})

if __name__ == "__main__":
    app.run()

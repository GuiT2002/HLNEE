import numpy as np
from keras.models import load_model


def preprocess_image(image):
    # Resizing the image so it matches the model training
    image = image.resize((32, 32))

    # Converting the image into an array
    image = np.array(image)

    # Normalizing the pixel values
    image = image / 255

    image = np.reshape(image, (1, 32, 32, 3))

    return image


# Predicting the image
def prediction(image):
    model = load_model('HLN Model v0.1.h5')
    prediction = model.predict(image)
    top_classes = prediction.argsort()[0][-5:][::-1]
    top_probs = prediction[0][top_classes]

    labels_list = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur',
                   'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                   'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                   'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                   'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                   'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
                   'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                   'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    class_labels = []

    for i in range(0, 5):
        class_labels.append(labels_list[top_classes[i]])

    return class_labels, top_probs



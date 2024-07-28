import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model


prototxt_path = 'deploy.prototxt'
model_path = 'mobilenet_iter_73000.caffemodel'


net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


car_model = load_model('car_color_model.keras')
gender_model = load_model('gender_classification_model.keras')


class_labels = np.load('car_class_labels.npy', allow_pickle=True)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def detect_objects(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    bounding_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Minimum confidence threshold
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["car", "person"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = CLASSES[idx]
                bounding_boxes.append((label, (startX, startY, endX - startX, endY - startY)))
    return bounding_boxes

def correct_bounding_box(box, image_shape):
    label, (startX, startY, width, height) = box
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(image_shape[1], startX + width)
    endY = min(image_shape[0], startY + height)
    corrected_box = (label, (startX, startY, endX - startX, endY - startY))
    return corrected_box

def detect_car_color(image, model):
    img_size = (128, 128)
    try:
        if image.size == 0:
            raise ValueError("Empty image for resizing")
        image = cv2.resize(image, img_size)
    except Exception as e:
        st.error(f"Error in resizing image: {e}")
        return None
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    car_color = class_labels[predicted_class]

    
    if car_color == 'red':
        car_color = 'blue'
    elif car_color == 'blue':
        car_color = 'red'

    return car_color

def count_cars_and_people(image, car_model, gender_model):
    car_count = 0
    male_count = 0
    female_count = 0
    car_colors = []

    bounding_boxes = detect_objects(image)

    for box in bounding_boxes:
        corrected_box = correct_bounding_box(box, image.shape)
        label, (x, y, w, h) = corrected_box
        if w <= 0 or h <= 0:
            st.warning(f"Skipping invalid bounding box: {corrected_box}")
            continue
        
        cropped_image = image[y:y+h, x:x+w]
        if cropped_image.size == 0:
            st.warning(f"Skipping empty cropped image for bounding box: {corrected_box}")
            continue

        if label == 'car':
            car_count += 1
            car_color = detect_car_color(cropped_image, car_model)
            if car_color is not None:
                car_colors.append(car_color)
        elif label == 'person':
            img_size = (128, 128)
            try:
                cropped_image = cv2.resize(cropped_image, img_size)
            except Exception as e:
                st.warning(f"Skipping invalid cropped image for bounding box: {corrected_box}")
                continue
            cropped_image = np.expand_dims(cropped_image, axis=0)
            cropped_image = cropped_image / 255.0
            prediction = gender_model.predict(cropped_image)
            if prediction < 0.5:
                male_count += 1
            else:
                female_count += 1

    return car_colors, car_count, male_count, female_count


st.title("Car Color and Number of Cars, Females, Males in a Traffic Detection")

uploaded_file = st.file_uploader("Choose a Traffic Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    car_colors, car_count, male_count, female_count = count_cars_and_people(image, car_model, gender_model)

    st.write(f"Car colors: {car_colors}")
    st.write(f"Number of cars: {car_count}")
    st.write(f"Number of males: {male_count}")
    st.write(f"Number of females: {female_count}")

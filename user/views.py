from django.shortcuts import render
from user.models import *
from django.contrib import messages
from django.conf import settings

import os
import numpy as np
import cv2
import pyttsx3
import time

from collections import Counter
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# LOAD MODEL

MODEL_PATH = os.path.join(settings.BASE_DIR, 'currency_classifier.h5')

try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Model load error:", e)
    model = None


# AUTHENTICATION

def userregister(request):
    return render(request, 'user/userregister.html')


def usersignup(request):
    if request.method == 'POST':
        Details = UserModel(
            username=request.POST.get('username'),
            email=request.POST.get('email'),
            password=request.POST.get('password'),
            mobile=request.POST.get('mobile'),
            city=request.POST.get('city'),
            state=request.POST.get('state')
        )
        Details.save()

    return render(request, 'user/userlogin.html')


def userloginaction(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        try:
            UserModel.objects.get(email=email, password=password)
            return render(request, "user/userhome.html")
        except:
            messages.success(request, "User not registered")

    return render(request, "user/userlogin.html")


def userhome(request):
    return render(request, 'user/userhome.html')


def userlogout(request):
    return render(request, 'user/userlogin.html')


# UTILITY

def find_repeating(arr):
    counts = Counter(arr)
    return max(counts, key=counts.get)


# MAIN

def userpredict(request):

    if model is None:
        return render(request, 'user/userhome.html', {'error': 'Model not loaded'})

    images_path = os.path.join(settings.MEDIA_ROOT, 'images')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Clear old images
    for file in os.listdir(images_path):
        os.remove(os.path.join(images_path, file))

    # CAMERA SETUP

    cam = cv2.VideoCapture(1)

    # Force higher resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cam.isOpened():
        return render(request, 'user/userhome.html', {'error': 'Camera not accessible'})

    print("Warming up camera...")

    # WARM-UP PHASE
    for _ in range(30):  # ~2 seconds
        cam.read()
        time.sleep(0.05)

    print("Capturing images...")

    # IMAGE CAPTURE
    captured_files = []

    for i in range(10):
        ret, frame = cam.read()
        if not ret:
            continue

        file_path = os.path.join(images_path, f'img{i}.jpg')

        # Save full resolution with max quality
        cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        captured_files.append(file_path)
        print(f"Captured frame {i}")

        time.sleep(0.2)

    cam.release()
    cv2.destroyAllWindows()

    if len(captured_files) == 0:
        return render(request, 'user/userhome.html', {'error': 'No images captured'})

    print("Running predictions...")

    # PREDICTION

    avg_pred = None

    for img_path in captured_files:
        try:
            # Resize ONLY here (not during capture)
            test_img = image.load_img(img_path, target_size=(224, 224))
            test_img = image.img_to_array(test_img) / 255.0
            test_img = np.expand_dims(test_img, axis=0)

            pred = model.predict(test_img, verbose=0)[0]

            if avg_pred is None:
                avg_pred = pred
            else:
                avg_pred += pred

        except Exception as e:
            print("Image error:", e)

    if avg_pred is None:
        return render(request, 'user/userhome.html', {'error': 'Prediction failed'})

    final_class = np.argmax(avg_pred)

    label_map = {
        0: '100',
        1: '200',
        2: '2000',
        3: '500',
        4: '50',
        5: '10',
        6: '20'
    }

    detected_note = label_map.get(final_class, "Unknown")

    print("Detected Note:", detected_note)

    # TEXT TO SPEECH

    try:
        engine = pyttsx3.init()
        engine.say(f"You have {detected_note} rupees note")
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("TTS Error:", e)

    return render(request, 'user/userhome.html', {'result': detected_note})
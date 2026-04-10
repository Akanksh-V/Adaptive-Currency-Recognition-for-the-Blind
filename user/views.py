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


# FAKE NOTE DETECTION

def verify_genuine(note_class, test_img_path):
    ref_dir = os.path.join(settings.BASE_DIR, 'assets', 'references')
    ref_path = os.path.join(ref_dir, f'{note_class}.jpg')
    
    if not os.path.exists(ref_path):
        print(f"Skipping fake check, no reference image found at {ref_path}")
        return True # Soft fallback if user hasn't provided a reference yet
        
    img1 = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return True
        
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return False
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Require at least 30 keypoint matches as a heuristic for "genuine"
    if len(matches) < 30:
        return False
    return True


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
    if not cam.isOpened():
        print("Camera at index 1 failed. Trying index 0...")
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera at index 0 failed. Trying index 2...")
        cam = cv2.VideoCapture(2)

    # Force higher resolution
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        return render(request, 'user/userhome.html', {'error': 'Camera not accessible. Please ensure your camera is connected and not being used by another application.'})

    print("Warming up camera...")

    # WARM-UP PHASE
    for _ in range(50):  # ~3 seconds
        cam.read()
        time.sleep(0.05)

    print("Capturing images...")

    # IMAGE CAPTURE
    captured_files = []

    for i in range(20):
        ret, frame = cam.read()
        if not ret:
            continue

        file_path = os.path.join(images_path, f'img{i}.jpg')

        # Save full resolution with max quality
        cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        captured_files.append(file_path)
        print(f"Captured frame {i}")

        time.sleep(0.1)

    cam.release()
    cv2.destroyAllWindows()

    print("Running predictions...")

    # PREDICTION
    predictions = []

    for img_path in captured_files:
        try:
            # Resize ONLY here (not during capture)
            test_img = image.load_img(img_path, target_size=(224, 224))
            test_img = image.img_to_array(test_img) / 255.0
            test_img = np.expand_dims(test_img, axis=0)

            pred = model.predict(test_img, verbose=0)[0]
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]

            # Only accept frames where it's at least 40% confident
            if confidence > 0.4:
                predictions.append(class_idx)

        except Exception as e:
            print("Image error:", e)

    # If predictions were found, we skip alignment complaints. 
    # If NO predictions were found, we try to give the user alignment feedback.
    if not predictions:
        # ALIGNMENT CHECK
        alignment_issues = []
        for img_path in captured_files:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            largest_area = 0
            best_cx = 0
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area > largest_area:
                    largest_area = area
                    best_cx = x + (w // 2)

            frame_width = img.shape[1]
            
            if largest_area < 80000: 
                alignment_issues.append("Check your frame, no note detected.")
            else:
                if best_cx < frame_width * 0.3:
                    alignment_issues.append("Move the note slightly to the right.")
                elif best_cx > frame_width * 0.7:
                    alignment_issues.append("Move the note slightly to the left.")
                else:
                    alignment_issues.append("Could not recognize the note clearly limit. Please try again.")
        
        most_common_issue = find_repeating(alignment_issues)
        if most_common_issue is None:
            most_common_issue = "Could not recognize the note clearly, please scan again."
            
        return render(request, 'user/userhome.html', {'error': most_common_issue})

    final_class = find_repeating(predictions)

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

    # FAKE NOTE VERIFICATION
    if detected_note != "Unknown":
        is_genuine = False
        for path in captured_files:
            if verify_genuine(detected_note, path):
                is_genuine = True
                break
                
        if not is_genuine:
            return render(request, 'user/userhome.html', {'error': 'Warning, this appears to be a Fake note.'})

    # TEXT TO SPEECH (Handled in frontend via Window.speechSynthesis to avoid macOS pyttsx3 server hang)
    
    return render(request, 'user/userhome.html', {'result': detected_note})
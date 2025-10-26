# =========================================================================
# 🔴 1. FULL SYSTEM SETUP & IMPORTS
# =========================================================================
import streamlit as st
import cv2
import numpy as np
import os
import pickle
import bz2
import shutil
import urllib.request
from PIL import Image
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern

# =========================================================================
# --- MODELS FOLDER ---
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

DLIB_LANDMARK_PATH = os.path.join(MODEL_FOLDER, "shape_predictor_68_face_landmarks.dat")
FACE_DB_PATH = os.path.join(MODEL_FOLDER, "face_user_db_multi.pkl")
HAND_DB_PATH = os.path.join(MODEL_FOLDER, "hand_user_db1.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_FOLDER, "gesture_classifier.pkl")
GESTURE_COMMANDS_PATH = os.path.join(MODEL_FOLDER, "gesture_commands.pkl")
CHARACTER_PATH = os.path.join(MODEL_FOLDER, "character.png")

# =========================================================================
# 🔹 STREAMLIT CACHING FOR MODELS
@st.cache_resource(show_spinner=False)
def load_dlib_models():
    import dlib
    if not os.path.exists(DLIB_LANDMARK_PATH):
        st.info("Downloading shape predictor (~100MB)...")
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = DLIB_LANDMARK_PATH + ".bz2"
        urllib.request.urlretrieve(url, bz2_path)
        with bz2.open(bz2_path, "rb") as f_in, open(DLIB_LANDMARK_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(bz2_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
    return detector, predictor

@st.cache_resource(show_spinner=False)
def load_fer_model():
    from fer import FER
    return FER(mtcnn=False)

@st.cache_resource(show_spinner=False)
def load_deepface_model():
    from deepface import DeepFace
    model = DeepFace.build_model("Facenet512")
    return model

@st.cache_resource(show_spinner=False)
def load_mediapipe_hands():
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    return hands

# =========================================================================
# 🔹 INITIALIZE MODELS (LAZY LOADING)
dlib_detector, dlib_predictor = load_dlib_models()
fer_detector = load_fer_model()
deepface_model = load_deepface_model()
hands = load_mediapipe_hands()

# =========================================================================
# 🔹 LOAD DATABASES
face_user_db = {}
if os.path.exists(FACE_DB_PATH):
    with open(FACE_DB_PATH, "rb") as f: face_user_db = pickle.load(f)

hand_user_db = {}
if os.path.exists(HAND_DB_PATH):
    with open(HAND_DB_PATH, "rb") as f: hand_user_db = pickle.load(f)

gesture_clf = None
gesture_commands = {}
try:
    with open(CLASSIFIER_PATH, "rb") as f: gesture_clf = pickle.load(f)
    with open(GESTURE_COMMANDS_PATH, "rb") as f: gesture_commands = pickle.load(f)
except:
    pass

# =========================================================================
# 🛠 HELPER FUNCTIONS
def capture_webcam():
    img_file_buffer = st.camera_input("Capture Image")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        image_cv = np.array(image)[:,:,::-1]  # RGB to BGR
        return image_cv, img_file_buffer
    return None, None

def extract_iris_dlib(gray, landmarks, eye_points):
    xs = [landmarks.part(p).x for p in eye_points]
    ys = [landmarks.part(p).y for p in eye_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    eye_img = gray[y_min:y_max, x_min:x_max]
    if eye_img.size == 0: return None
    eye_img = cv2.resize(eye_img, (64,64))
    return eye_img

def get_iris_features(img_path):
    import dlib
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray)
    feats = []
    for face in faces:
        landmarks = dlib_predictor(gray, face)
        for eye_points in [range(36,42), range(42,48)]:
            eye_img = extract_iris_dlib(gray, landmarks, eye_points)
            if eye_img is not None:
                lbp = local_binary_pattern(eye_img, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
                hist = hist.astype("float"); hist /= (hist.sum() + 1e-6)
                feats.append(hist.reshape(1, -1))
    return np.mean(feats, axis=0) if feats else None

def extract_hand_landmarks(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks: return None
    lm = results.multi_hand_landmarks[0]
    points = np.array([(l.x, l.y) for l in lm.landmark])
    center = np.mean(points, axis=0)
    points -= center
    norm = np.linalg.norm(points)
    if norm > 0: points /= norm
    return points.flatten()

def speak_with_character(text):
    if text:
        tts = gTTS(text)
        tts.save("output.mp3")
        audio_file = open("output.mp3", "rb")
        if os.path.exists(CHARACTER_PATH):
            char_img = Image.open(CHARACTER_PATH)
            st.image(char_img, caption="🤖 Speaking...", width=200)
        st.audio(audio_file.read(), format="audio/mp3")
        st.success(text)

# =========================================================================
# 🎯 AUTHENTICATION FUNCTIONS
def authenticate_face(captured_image_path, username, threshold=0.35):
    if username not in face_user_db: return False
    from deepface import DeepFace
    try:
        cap_emb = DeepFace.represent(captured_image_path, model_name="Facenet512", model=deepface_model, enforce_detection=True)[0]["embedding"]
    except:
        return False
    best_similarity = -1
    for emb in face_user_db[username]:
        cos_sim = np.dot(emb, cap_emb) / (np.linalg.norm(emb) * np.linalg.norm(cap_emb))
        if cos_sim > best_similarity:
            best_similarity = cos_sim
    return best_similarity >= threshold

def authenticate_iris(captured_image_path, username, threshold=0.6):
    iris_enrolled_feat = np.ones((1, 9)) * 0.111 
    current_feat = get_iris_features(captured_image_path)
    if current_feat is None: return False
    sim = cosine_similarity(iris_enrolled_feat, current_feat)[0][0]
    return sim >= threshold

def authenticate_hand(frame, username, threshold=2.5):
    if username not in hand_user_db: return False
    emb = extract_hand_landmarks(frame)
    if emb is None: return False
    distances = [np.linalg.norm(emb - e) for e in hand_user_db[username]]
    best_dist = min(distances)
    return best_dist < threshold

def master_multi_modal_authenticate(username, frame_path):
    frame = cv2.imread(frame_path)
    auth_face = authenticate_face(frame_path, username)
    auth_iris = authenticate_iris(frame_path, username)
    auth_hand = authenticate_hand(frame, username)
    auth_results = {"Face": auth_face, "Iris": auth_iris, "Hand": auth_hand}
    successful_matches = sum(auth_results.values())
    if successful_matches >= 2: 
        status = f"✅ AUTHENTICATION PASSED! ({successful_matches}/3 modalities matched)"
        return True, status
    else:
        status = f"❌ AUTHENTICATION FAILED. ({successful_matches}/3 modalities matched)"
        return False, status

# =========================================================================
# 4. STREAMLIT UI
st.title("🔐 Multi-Modal Biometric Authentication System")
username = st.text_input("Enter User Name")
menu = st.selectbox("Select Action", ["Enrollment", "Authentication"])

if menu == "Enrollment":
    st.subheader("User Enrollment")
    image_cv, img_file = capture_webcam()
    if st.button("Register User") and username and image_cv is not None:
        temp_face_path = os.path.join(MODEL_FOLDER, f"{username}_face.jpg")
        cv2.imwrite(temp_face_path, image_cv)
        from deepface import DeepFace
        try:
            emb = DeepFace.represent(temp_face_path, model_name="Facenet512", model=deepface_model, enforce_detection=True)[0]["embedding"]
            face_user_db[username] = [emb]
            with open(FACE_DB_PATH, "wb") as f: pickle.dump(face_user_db, f)
            st.success("✅ Face registered")
        except:
            st.error("❌ Face registration failed")
        emb_hand = extract_hand_landmarks(image_cv)
        if emb_hand is not None:
            hand_user_db[username] = [emb_hand]
            with open(HAND_DB_PATH, "wb") as f: pickle.dump(hand_user_db, f)
            st.success("✅ Hand registered")
        else:
            st.warning("⚠ Hand not detected")

elif menu == "Authentication":
    st.subheader("User Authentication")
    image_cv, img_file = capture_webcam()
    if st.button("Authenticate") and username and image_cv is not None:
        temp_path = os.path.join(MODEL_FOLDER, f"{username}_auth.jpg")
        cv2.imwrite(temp_path, image_cv)
        success, message = master_multi_modal_authenticate(username, temp_path)
        st.write(message)
        if success:
            speak_with_character("Authentication successful!")

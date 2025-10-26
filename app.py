import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import pickle
from deepface import DeepFace
import mediapipe as mp
from gtts import gTTS

# ======================
# --- Paths / Models ---
# ======================
MODEL_FOLDER = "model"
os.makedirs(MODEL_FOLDER, exist_ok=True)

FACE_DB_PATH = os.path.join(MODEL_FOLDER, "face_user_db.pkl")
HAND_DB_PATH = os.path.join(MODEL_FOLDER, "hand_user_db.pkl")
GESTURE_CLF_PATH = os.path.join(MODEL_FOLDER, "gesture_clf.pkl")
GESTURE_CMDS_PATH = os.path.join(MODEL_FOLDER, "gesture_commands.pkl")
CHARACTER_PATH = os.path.join(MODEL_FOLDER, "doraemon.png")  # Doraemon image

# ======================
# --- Initialization ---
# ======================
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

face_user_db, hand_user_db, gesture_clf, gesture_commands = {}, {}, None, {}

if os.path.exists(FACE_DB_PATH):
    with open(FACE_DB_PATH, "rb") as f: face_user_db = pickle.load(f)
if os.path.exists(HAND_DB_PATH):
    with open(HAND_DB_PATH, "rb") as f: hand_user_db = pickle.load(f)
if os.path.exists(GESTURE_CLF_PATH):
    with open(GESTURE_CLF_PATH, "rb") as f: gesture_clf = pickle.load(f)
if os.path.exists(GESTURE_CMDS_PATH):
    with open(GESTURE_CMDS_PATH, "rb") as f: gesture_commands = pickle.load(f)

# ======================
# --- Helper Functions ---
# ======================
def extract_hand_landmarks(frame):
    results = hands_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks: 
        return None
    lm = results.multi_hand_landmarks[0]
    points = np.array([(l.x, l.y) for l in lm.landmark])
    points -= np.mean(points, axis=0)
    norm = np.linalg.norm(points)
    if norm > 0: points /= norm
    return points.flatten()

def speak(text):
    if text:
        tts = gTTS(text)
        tts.save("output.mp3")
        st.audio("output.mp3")
        if os.path.exists(CHARACTER_PATH):
            st.image(CHARACTER_PATH, caption="Doraemon says:", width=200)

# ======================
# --- Face Authentication ---
# ======================
def authenticate_face(img_path, username, threshold=0.35):
    if username not in face_user_db: return False
    try:
        emb = DeepFace.represent(img_path, model_name="OpenFace", enforce_detection=True)[0]["embedding"]
    except: return False
    best = max(np.dot(emb_db, emb)/(np.linalg.norm(emb_db)*np.linalg.norm(emb)) for emb_db in face_user_db[username])
    return best >= threshold

# ======================
# --- Hand Authentication ---
# ======================
def authenticate_hand(frame, username, threshold=2.5):
    if username not in hand_user_db: return False
    emb = extract_hand_landmarks(frame)
    if emb is None: return False
    best_dist = min([np.linalg.norm(emb-e) for e in hand_user_db[username]])
    return best_dist < threshold

# ======================
# --- Multi-Modal Auth ---
# ======================
def master_authenticate(username, frame_path):
    frame = cv2.imread(frame_path)
    f = authenticate_face(frame_path, username)
    h = authenticate_hand(frame, username)
    matches = sum([f,h])
    status = f"✅ AUTHENTICATION PASSED! ({matches}/2 matched)" if matches>=1 else f"❌ AUTHENTICATION FAILED ({matches}/2 matched)"
    return matches>=1, status

# ======================
# --- Streamlit UI ---
# ======================
st.title("🔐 Doraemon Multi-Modal Authentication")

username = st.text_input("Enter User Name")
menu = st.selectbox("Select Action", ["Enrollment","Authentication"])

# ----------------------
# Enrollment
# ----------------------
if menu=="Enrollment":
    st.subheader("Enroll User")
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg","png"])
    if st.button("Enroll") and uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)[:,:,::-1]
        path = f"temp_{username}.jpg"
        cv2.imwrite(path, frame)

        # --- Face ---
        emb = DeepFace.represent(path, model_name="OpenFace", enforce_detection=True)[0]["embedding"]
        face_user_db[username] = [emb]
        with open(FACE_DB_PATH,"wb") as f: pickle.dump(face_user_db,f)

        # --- Hand ---
        hand_emb = extract_hand_landmarks(frame)
        if hand_emb is not None:
            hand_user_db[username] = [hand_emb]
            with open(HAND_DB_PATH,"wb") as f: pickle.dump(hand_user_db,f)

        st.success(f"✅ User {username} enrolled!")

# ----------------------
# Authentication
# ----------------------
elif menu=="Authentication":
    st.subheader("Authenticate User")
    uploaded_file = st.file_uploader("Upload Image for Authentication", type=["jpg","png"])
    if st.button("Authenticate") and uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)[:,:,::-1]
        path = f"temp_auth_{username}.jpg"
        cv2.imwrite(path, frame)
        success, msg = master_authenticate(username, path)
        st.write(msg)
        if success:
            speak("Authentication successful!")

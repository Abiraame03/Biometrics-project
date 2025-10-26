import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import pickle
from deepface import DeepFace
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
import dlib
import mediapipe as mp
from gtts import gTTS

# ======================
# --- PATHS / MODELS ---
# ======================
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

DLIB_PATH = os.path.join(MODEL_FOLDER, "shape_predictor_68_face_landmarks.dat")
FACE_DB_PATH = os.path.join(MODEL_FOLDER, "face_user_db.pkl")
HAND_DB_PATH = os.path.join(MODEL_FOLDER, "hand_user_db.pkl")
GESTURE_CLF_PATH = os.path.join(MODEL_FOLDER, "gesture_clf.pkl")
GESTURE_CMDS_PATH = os.path.join(MODEL_FOLDER, "gesture_commands.pkl")
CHARACTER_PATH = os.path.join(MODEL_FOLDER, "character.png")

# ======================
# --- DOWNLOAD DLIB IF MISSING ---
# ======================
import urllib.request, bz2, shutil
if not os.path.exists(DLIB_PATH):
    st.info("Downloading DLIB shape predictor...")
    url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, f"{DLIB_PATH}.bz2")
    with bz2.open(f"{DLIB_PATH}.bz2", "rb") as f_in, open(DLIB_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(f"{DLIB_PATH}.bz2")
    st.success("✅ DLIB downloaded")

# ======================
# --- INITIALIZATION ---
# ======================
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(DLIB_PATH)
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ======================
# --- LOAD PICKLED MODELS ---
# ======================
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
# --- FEATURES ---
# ======================
def extract_iris(gray, landmarks, eye_points):
    xs = [landmarks.part(p).x for p in eye_points]
    ys = [landmarks.part(p).y for p in eye_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    eye_img = gray[y_min:y_max, x_min:x_max]
    if eye_img.size == 0: return None
    return cv2.resize(eye_img, (64,64))

def get_iris_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray)
    feats = []
    for face in faces:
        landmarks = dlib_predictor(gray, face)
        for eye_points in [range(36,42), range(42,48)]:
            eye_img = extract_iris(gray, landmarks, eye_points)
            if eye_img is not None:
                lbp = local_binary_pattern(eye_img, 8,1,"uniform")
                hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,10), range=(0,9))
                hist = hist.astype("float"); hist /= (hist.sum()+1e-6)
                feats.append(hist.reshape(1,-1))
    return np.mean(feats,axis=0) if feats else None

def extract_hand_landmarks(frame):
    results = hands_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks: return None
    lm = results.multi_hand_landmarks[0]
    points = np.array([(l.x,l.y) for l in lm.landmark])
    points -= np.mean(points,axis=0)
    norm = np.linalg.norm(points)
    if norm>0: points/=norm
    return points.flatten()

def speak(text):
    if text:
        tts = gTTS(text)
        tts.save("output.mp3")
        st.audio("output.mp3")

# ======================
# --- AUTHENTICATION ---
# ======================
def authenticate_face(img_path, username, threshold=0.35):
    if username not in face_user_db: return False
    try:
        emb = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=True)[0]["embedding"]
    except: return False
    best = max(np.dot(emb_db, emb)/(np.linalg.norm(emb_db)*np.linalg.norm(emb)) for emb_db in face_user_db[username])
    return best>=threshold

def authenticate_iris(img_path, username, threshold=0.6):
    iris_feat = get_iris_features(img_path)
    if iris_feat is None: return False
    enrolled_feat = np.ones((1,9))*0.111
    sim = cosine_similarity(enrolled_feat, iris_feat)[0][0]
    return sim>=threshold

def authenticate_hand(frame, username, threshold=2.5):
    if username not in hand_user_db: return False
    emb = extract_hand_landmarks(frame)
    if emb is None: return False
    best_dist = min([np.linalg.norm(emb-e) for e in hand_user_db[username]])
    return best_dist<threshold

def master_authenticate(username, frame_path):
    frame = cv2.imread(frame_path)
    f = authenticate_face(frame_path, username)
    i = authenticate_iris(frame_path, username)
    h = authenticate_hand(frame, username)
    matches = sum([f,i,h])
    status = f"✅ AUTHENTICATION PASSED! ({matches}/3 matched)" if matches>=2 else f"❌ AUTHENTICATION FAILED ({matches}/3 matched)"
    return matches>=2, status

# ======================
# --- STREAMLIT UI ---
# ======================
st.title("🔐 Multi-Modal Biometric Auth System")
username = st.text_input("Enter User Name")
menu = st.selectbox("Select Action", ["Enrollment","Authentication"])

if menu=="Enrollment":
    st.subheader("Enroll User")
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg","png"])
    if st.button("Enroll") and uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)[:,:,::-1]
        path = f"temp_{username}.jpg"
        cv2.imwrite(path, frame)
        # Face Embedding
        emb = DeepFace.represent(path, model_name="Facenet512", enforce_detection=True)[0]["embedding"]
        face_user_db[username] = [emb]
        with open(FACE_DB_PATH,"wb") as f: pickle.dump(face_user_db,f)
        # Hand embedding
        hand_emb = extract_hand_landmarks(frame)
        if hand_emb is not None:
            hand_user_db[username] = [hand_emb]
            with open(HAND_DB_PATH,"wb") as f: pickle.dump(hand_user_db,f)
        st.success(f"✅ User {username} enrolled!")

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

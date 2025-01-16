import cv2
import face_recognition
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

# Prepare une liste pour stocker les visages et noms qu'on va ajouter
if "known_face_encodings" not in st.session_state:
    st.session_state["known_face_encodings"] = []
    st.session_state["known_face_names"] = []

# Affiche le titre de l'appli
st.title("Application de Reconnaissance Faciale")

# Partie pour ajouter un visage à reconnaître
st.subheader("Configurer les visages connus")

uploaded_image = st.file_uploader("Charger une image de visage", type=["jpg", "jpeg", "png"])
name = st.text_input("Nom associé au visage")

if st.button("Ajouter à la base des visages connus"):
    if uploaded_image and name:
        try:
            # Charge l'image et récupère son encodage
            pil_image = Image.open(uploaded_image)
            face_image = np.array(pil_image)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            # Ajoute l'encodage et le nom à la liste
            st.session_state["known_face_encodings"].append(face_encoding)
            st.session_state["known_face_names"].append(name)

            st.success(f"Visage de {name} ajouté avec succès !")
        except IndexError:
            st.error("Pas de visage détecté sur l'image, essaie une autre photo.")
    else:
        st.warning("Charge une image et entre un nom avant de valider.")

# Bouton pour vider la liste des visages
if st.button("Vider la base des visages connus"):
    st.session_state["known_face_encodings"].clear()
    st.session_state["known_face_names"].clear()
    st.success("La base des visages connus est maintenant vide.")

# Affiche les visages déjà configurés
if st.session_state["known_face_names"]:
    st.subheader("Visages connus configurés")
    for idx, label in enumerate(st.session_state["known_face_names"]):
        st.write(f"{idx + 1}. {label}")

# Partie pour reconnaître des visages
st.subheader("Reconnaissance Faciale")

# Choix entre webcam ou fichier vidéo
video_source = st.selectbox("Source vidéo", ["Webcam", "Fichier vidéo"])

if video_source == "Fichier vidéo":
    video_file = st.file_uploader("Charger un fichier vidéo", type=["mp4", "avi", "mov"])
    if video_file:
        # Enregistre la vidéo temporairement pour qu'elle puisse être lue
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            tmp_video_path = tmp_file.name

# Boutons pour démarrer ou arrêter la reconnaissance
start_recognition = st.button("Démarrer la reconnaissance")
stop_recognition = st.button("Arrêter la reconnaissance")

# Zone où on affichera la vidéo
stframe = st.empty()

# Configure la source vidéo
video_capture = None
if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)
elif video_source == "Fichier vidéo" and video_file is not None:
    video_capture = cv2.VideoCapture(tmp_video_path)

# Gestion des états de démarrage/arrêt
if start_recognition and video_capture:
    st.session_state["running"] = True

if stop_recognition:
    st.session_state["running"] = False

# Boucle pour analyser la vidéo en temps réel
if st.session_state.get("running", False) and video_capture:
    while st.session_state["running"] and video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.error("Impossible de lire la vidéo.")
            break

        # Passe l'image en RGB pour qu'elle soit compatible avec face_recognition
        rgb_frame = frame[:, :, ::-1]

        # Détecte les visages dans l'image
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Regarde si ce visage correspond à quelqu'un qu'on connaît
            matches = face_recognition.compare_faces(st.session_state["known_face_encodings"], face_encoding)
            name = "Inconnu"

            # Si on trouve une correspondance, récupère le nom
            face_distances = face_recognition.face_distance(st.session_state["known_face_encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches and matches[best_match_index]:
                name = st.session_state["known_face_names"][best_match_index]

            # Dessine un rectangle autour du visage et ajoute le nom
            rectangle_color = (0, 0, 255)  # Rouge
            text_color = (255, 255, 255)  # Blanc
            font_scale = 1.5  # Taille du texte
            rectangle_thickness = 3  # Épaisseur du cadre

            cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, rectangle_thickness)
            cv2.rectangle(frame, (left, bottom - 50), (right, bottom), rectangle_color, cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (left + 10, bottom - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                2,
            )

        # Affiche la vidéo avec les annotations
        stframe.image(frame, channels="BGR")

    # Libère les ressources une fois terminé
    video_capture.release()
    stframe.empty()

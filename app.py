import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# --------------------------
# Load model & MediaPipe
# --------------------------

@st.cache_resource
def load_model():
    with open("model.p", "rb") as f:
        model_dict = pickle.load(f)
    return model_dict["model"]

@st.cache_resource
def get_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    return hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Same mapping as in inference_classifier.py
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}


# --------------------------
# Core prediction function
# (matches your inference logic)
# --------------------------

def predict_from_bgr(image_bgr, model, hands):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = image_bgr.shape
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    annotated = image_bgr.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect x, y lists
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Create normalized feature vector (length 42)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Only predict if vector is complete (21*2=42)
        if len(data_aux) == 42:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            label_index = int(prediction[0])
            predicted_char = labels_dict.get(label_index, "?")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(
                annotated,
                predicted_char,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

            return predicted_char, annotated

    # No hand or incomplete landmarks
    return None, annotated


# --------------------------
# Streamlit UI
# --------------------------

def main():
    st.set_page_config(page_title="Sign Language Detector", layout="wide")
    st.title("🖐 Sign Language Detector (A–Z)")

    st.markdown(
        """
        This app uses your trained **RandomForest** sign language classifier  
        built with **MediaPipe Hands + OpenCV** to recognize letters A–Z from a single hand sign.
        """
    )

    model = load_model()
    hands = get_hands()

    # For building a small word/sentence
    if "sentence" not in st.session_state:
        st.session_state["sentence"] = ""

    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio(
        "Choose how to provide an image:",
        ["Use Camera", "Upload Image"]
    )

    st.sidebar.subheader("Spelled Text")
    st.sidebar.text_area(
        "Predicted letters:",
        value=st.session_state["sentence"],
        height=100
    )

    col1, col2 = st.columns(2)

    if input_mode == "Use Camera":
        with col1:
            st.subheader("Camera Input")
            img_file = st.camera_input("Show a hand sign and capture a photo")

        if img_file is not None:
            pil_image = Image.open(img_file)
            img_rgb = np.array(pil_image)

            # Handle possible RGBA
            if img_rgb.shape[-1] == 4:
                img_rgb = img_rgb[:, :, :3]

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            pred_char, annotated = predict_from_bgr(img_bgr, model, hands)

            with col2:
                st.subheader("Prediction")
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                    caption="Detected hand landmarks"
                )

                if pred_char is not None:
                    st.success(f"Predicted Letter: **{pred_char}**")
                    a, b = st.columns(2)
                    with a:
                        if st.button("➕ Add letter to text"):
                            st.session_state["sentence"] += pred_char
                    with b:
                        if st.button("🧹 Clear text"):
                            st.session_state["sentence"] = ""
                else:
                    st.warning("No hand detected or not enough landmarks to predict.")

    else:  # Upload Image
        with col1:
            st.subheader("Upload Image")
            uploaded = st.file_uploader(
                "Upload an image containing a single hand sign",
                type=["jpg", "jpeg", "png"]
            )

        if uploaded is not None:
            pil_image = Image.open(uploaded)
            img_rgb = np.array(pil_image)

            if img_rgb.shape[-1] == 4:
                img_rgb = img_rgb[:, :, :3]

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            pred_char, annotated = predict_from_bgr(img_bgr, model, hands)

            with col2:
                st.subheader("Prediction")
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                    caption="Detected hand landmarks"
                )

                if pred_char is not None:
                    st.success(f"Predicted Letter: **{pred_char}**")
                    a, b = st.columns(2)
                    with a:
                        if st.button("➕ Add letter to text"):
                            st.session_state["sentence"] += pred_char
                    with b:
                        if st.button("🧹 Clear text"):
                            st.session_state["sentence"] = ""
                else:
                    st.warning("No hand detected or not enough landmarks to predict.")


if __name__ == "__main__":
    main()

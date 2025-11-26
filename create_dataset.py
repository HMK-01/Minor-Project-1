# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []

#         x_ = []
#         y_ = []

#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#             data.append(data_aux)
#             labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()

import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip non-directories like .gitignore
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)

        # Skip unreadable or broken images
        if img is None:
            print(f"⚠️ Could not read image: {img_full_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         for landmark in hand_landmarks.landmark:
        #             x_.append(landmark.x)
        #             y_.append(landmark.y)

        #         for landmark in hand_landmarks.landmark:
        #             data_aux.append(landmark.x - min(x_))
        #             data_aux.append(landmark.y - min(y_))

        #     data.append(data_aux)
        #     labels.append(dir_)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            # ✅ Only add if data_aux is complete (21 landmarks × 2 = 42 values)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"⚠️ Skipping image with incomplete landmarks: {img_full_path}")


# Save the data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Data collection completed and saved to 'data.pickle'")


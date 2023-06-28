import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

# Define the garbage classes and corresponding labels
classes = ["cardboard", "glass", "metal", "paper", "plastic"]
label_map = {label: i for i, label in enumerate(classes)}

# Load the trained model
model = tf.keras.models.load_model("model/best_model_2.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(frame / 255.0, axis=0)
    preds = model.predict(img)[0]
    pred_idx = np.argmax(preds)
    label = classes[pred_idx]
    confidence = preds[pred_idx]

    # Draw the predicted class label and confidence score on the frame
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Garbage Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
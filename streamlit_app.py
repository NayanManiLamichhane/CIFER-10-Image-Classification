
import streamlit as st
import numpy as np
import keras
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    layout="wide"
)

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@st.cache_resource
def load_model():
    return keras.models.load_model(
        "cifar10_model_patched.keras",
        compile=False
    )

st.title("CIFAR-10 Image Classifier")
st.write(
    "Upload an image and let the model predict which "
    "of the 10 CIFAR-10 classes it belongs to."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Your image")
        st.image(image, use_container_width=True)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):
            # CIFAR-10 input preprocessing
            resized_image = image.resize((32, 32))
            array = np.asarray(
                resized_image,
                dtype=np.float32
            ) / 255.0
            array = np.expand_dims(array, axis=0)

            model = load_model()

            # The saved model expects a batch of 32 images.
            # Repeat the image to match its fixed input shape.
            batch = np.repeat(array, 32, axis=0)

            predictions = model.predict(
                batch,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])

        with right:
            st.subheader("Prediction")

            st.success(
                f"Predicted class: "
                f"{CLASS_NAMES[predicted_index]}"
            )

            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Scores for all classes")

            for name, score in sorted(
                zip(CLASS_NAMES, predictions),
                key=lambda item: item[1],
                reverse=True
            ):
                st.write(
                    f"**{name}** — {score * 100:.2f}%"
                )
                st.progress(
                    float(np.clip(score, 0.0, 1.0))
                )




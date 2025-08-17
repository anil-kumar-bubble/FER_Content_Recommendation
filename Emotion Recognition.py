import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
import csv

# Function to predict emotions from the given image
def predict_emotions(image):
    # Load model architecture from JSON file
    json_file = open(r'E:\Major project\fermodel best\fermodel2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load model weights
    model.load_weights(r'E:\Major project\fermodel best\fermodel28199.weights.h5')

    # Load face cascade classifier
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    #top_emotions = []
    detected_emotions = []
    probabilities = []
    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + h, x:x + w]  # Extract face ROI
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.expand_dims(roi_gray, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)  # Add channel dimension
        img_pixels = img_pixels.astype('float32')
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        top_indices = np.argsort(predictions)[0][-2:][::-1]  # Indices of top two predicted emotions

        # If the second highest predicted emotion is neutral, happiness, or surprise, skip it
        if emotions[top_indices[1]] in ['neutral', 'happiness', 'surprise']:
            top_indices = top_indices[:1]  # Take only the highest predicted emotion

        detected_emotions.append([emotions[idx] for idx in top_indices])
        probabilities.append([predictions[0][idx] for idx in top_indices])

    return detected_emotions, probabilities, image

# Function to write data to CSV file and display in table format
def print_content(emotion, color):
    emotion_lower = emotion.lower() if isinstance(emotion, str) else str(emotion)
    file_paths = {
        'anger': {
            'movies': r"E:\Major project\Content dataset\anger_movies.csv",
            'books': r"E:\Major project\Content dataset\anger_books.csv",
            'music': r"E:\Major project\Content dataset\anger_music.csv"
        },
        'disgust': {
            'movies': r"E:\Major project\Content dataset\disgust_movies.csv",
            'books': r"E:\Major project\Content dataset\disgust_books.csv",
            'music': r"E:\Major project\Content dataset\disgust_music.csv"
        },
        'fear': {
            'movies': r"E:\Major project\Content dataset\fear_movies.csv",
            'books': r"E:\Major project\Content dataset\fear_books.csv",
            'music': r"E:\Major project\Content dataset\fear_music.csv"
        },
        'happiness': {
            'movies': r"E:\Major project\Content dataset\happiness_movies.csv",
            'books': r"E:\Major project\Content dataset\happiness_books.csv",
            'music': r"E:\Major project\Content dataset\happiness_music.csv"
        },
        'neutral': {
            'movies': r"E:\Major project\Content dataset\neutral_movies.csv",
            'books': r"E:\Major project\Content dataset\neutral_books.csv",
            'music': r"E:\Major project\Content dataset\neutral_music.csv"
        },
        'sadness': {
            'movies': r"E:\Major project\Content dataset\sadness_movies.csv",
            'books': r"E:\Major project\Content dataset\sadness_books.csv",
            'music': r"E:\Major project\Content dataset\sadness_music.csv"
        },
        'surprise': {
            'movies': r"E:\Major project\Content dataset\surprise_movies.csv",
            'books': r"E:\Major project\Content dataset\surprise_books.csv",
            'music': r"E:\Major project\Content dataset\surprise_music.csv"
        }
    }

    for category, path in file_paths[emotion_lower].items():
        with open(path, 'r') as file:
            reader = csv.reader(file)
            contents = list(reader)

        if contents:
            st.markdown(f"**_{category.capitalize()} for the {emotion_lower} emotion_**")
            st.table(contents)  # Print the first row in bold
            #st.table(contents[1:])
# Main function to create the Streamlit app
def main():
    st.title('Facial Emotion Recognition')

    if st.button('Take Photo'):
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        if ret:
            emotions, probabilities, image = predict_emotions(frame)

            # If multiple emotions are detected, print the content for both
            if len(emotions) > 1:
                st.image(image, channels="BGR", caption='Detected Emotions with Bounding Boxes')

                st.markdown("### Detected Emotions and Probabilities:")
                for emotion, prob in zip(emotions, probabilities):
                    st.write(f"- **{emotion.capitalize()}**: {prob:.2f}")

                st.markdown("### Content for Detected Emotions:")
                for emotion in emotions:
                    print_content(emotion, color='yellow')
            else:
                if len(emotions) > 0:
                    emotion = emotions[0]
                    st.image(image, channels="BGR", caption=f'Detected Emotion: {emotion}')

                    st.markdown("### Detected Emotion and Probability:")
                    prob = probabilities[0]
                    st.write("### Predicted Emotion:")
                    if len(emotion) > 1:
                        for emotion1, probability1 in zip(emotion, prob):
                            st.write(f"- **{emotion1.capitalize()}**: {probability1:.2f}")
                            st.markdown("### Content for Detected Emotion:")
                            print_content(emotion1, color='yellow')
                    elif len(emotion) == 1:
                        st.write(f"- **{emotion[0].capitalize()}**: {prob[0]:.2f}")
                        st.markdown("### Content for Detected Emotion:")
                        print_content(emotion[0], color='yellow')
                else:
                    st.error('Failed to capture image.')

        cap.release()

if __name__ == '__main__':
    main()

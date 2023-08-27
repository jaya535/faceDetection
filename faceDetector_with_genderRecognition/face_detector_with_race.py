from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
from PIL import Image
import numpy as np
import streamlit as st
import cv2

@st.cache_data(persist="disk")
def detect_attribs(face):
    """
    Function that detects gender, race and emotion for the detected face.
    """
    results=DeepFace.analyze(face,actions=("gender","race","emotion"),enforce_detection=False)
    results=results[0]
    gender=results['dominant_gender']
    race=results['dominant_race']
    emotion=results['dominant_emotion']
    return gender,race,emotion

@st.cache_data(persist="disk")
def mark_faces(image,faces):
    """
    Function returns the gender, race and emotion of all the detected faces and returns them in the form of arrays. 
    """
    face_images=[]
    genders=[]
    races=[]
    emotions=[]
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        extracted_face=image[y1:y2,x1:x2]
        gender,race,emotion=detect_attribs(extracted_face)
        face_images.append(extracted_face)
        genders.append(gender)
        races.append(race)
        emotions.append(emotion)
        image=cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,255,0),thickness=2)
    return image,face_images,genders,races,emotions

upload_flag=0
faces_in_each_row=4
st.set_page_config(layout="wide",initial_sidebar_state='expanded')
st.markdown("<h1 style='text-align:center;background-color:orange'> Face Detector with Gender Recognition and Emotion Classificaton </h1>",unsafe_allow_html=True)

#sidebar
st.sidebar.markdown("<h1 style='text-align:center'><u><b> Developer </b></u></h1>",unsafe_allow_html=True)
with st.sidebar:
    st.image("Images/me.jpg")
    with st.expander("About Me"):
        st.write("""
            Hello, I'm a Data Scientist with 2 years of experience in data analysis, machine learning, and deep learning. I have a strong background in Computer Science which allows me to apply a wide range of techniques to solve complex business problems.

            I have a passion for understanding and solving complex data problems, and I pride myself on my ability to communicate technical concepts to non-technical stakeholders. I believe that the key to success in data science is to approach problems with curiosity, rigor, and creativity, and to continuously learn and adapt to new technologies and techniques.
        """)
    st.subheader("Social Links")
    col1,col2,col3=st.columns(3)
    col1.markdown("<a href='https://www.linkedin.com/in/nagasai-biginepalli-64648a146/'>Linkedin</a>",unsafe_allow_html=True)
    col2.markdown("<a href='https://github.com/Nagasai524'>Github</a>",unsafe_allow_html=True)
    col3.markdown("<a href='mailto:www.biginepallinagasai109@gmail.com'>Gmail</a>",unsafe_allow_html=True)

col1,col2=st.columns(2)
uploaded_image=col1.camera_input(label="Take a Photo")
col1.info("Please take the photo under good lighting conditions for better results.")
if uploaded_image is None:
    uploaded_image=col2.file_uploader(label="Upload Any Image with faces",type=['png','jpg','jpeg','webp'],accept_multiple_files=False)
    upload_flag=1
    with col2.expander("Sample image for testing"):
        st.image('Images/faces.jpeg')
st.snow()
if uploaded_image is not None:
    image=Image.open(uploaded_image)
    img_array=np.array(image)
    detector=MTCNN()
    faces=detector.detect_faces(img_array)
    if len(faces)==0 and not(upload_flag):
        st.error(":x: Captured Image did not have any faces in it. Try uploading an image that have faces. :x:")
        capture_flag=0
    elif len(faces)==0 and upload_flag:
        st.error(":x: Uploaded Image did not have any faces in it. Try capturing your live Photo using webcam. :x:")
    else:
        st.success(":heavy_check_mark: Image Uploaded Successfully. Detecting faces and attributes. :heavy_check_mark:")
        st.info("The execution time depends on the number of faces in the image. Please Wait!!!")
        with st.spinner("Processing"):
            processed_image,face_images,genders,races,emotions=mark_faces(img_array,faces)
            genders=['Male' if x=='Man' else 'Female' for x in genders]
            face_images=[cv2.resize(x, (200, 200),interpolation = cv2.INTER_LINEAR) for x in face_images]
            st.markdown("<h3 style='text-align:center'><u> Processeed Image </u></h3>",unsafe_allow_html=True)
            col1,col2,col3=st.columns([1,5,1])
            with col2:
                st.image(processed_image)
            st.markdown("<h3 style='text-align:center'><u> Detected Faces </u></h3>",unsafe_allow_html=True)
            for i in range(len(face_images)):
                if i%faces_in_each_row==0:
                    container=st.container()
                    col1,col2,col3,col4=container.columns(faces_in_each_row)
                    column_array=[col1,col2,col3,col4]
                with container:
                    with column_array[i%faces_in_each_row]:
                        st.image(face_images[i])
                        st.markdown("**Gender** : "+genders[i])
                        st.markdown("**Race** : "+races[i])
                        st.markdown("**Emotion** : "+emotions[i])
        st.balloons()
       
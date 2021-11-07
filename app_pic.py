import cv2
import streamlit as st
import os
import numpy as np
import pandas as pd
import face_recognition
import cv2


# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 6, 3))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title("Face Recognition Model")
    st.markdown("<h2>A Generic Deep Learning POC for Face Recognition</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared as a proof of concept of an artificial intelligence (AI) model to recognize the person from his face
                """)






st.subheader('How to use the model?')
''' 
You can use the model by following the below steps:

1- You can upload the photo of the person that you want the model to recognize

2- The model will detect the face/ faces in the image and identify them by numbers starting from 0 and those numbers will be listed in the list under the 'select face' section

3- By selecting the number of the face, the model tells you how much the face is similar to the faces in the databsse by showing the below table. Also, you have the option to add the person to the database if he/she doesn't exist in it

4- if you decide to to add the person to the database, you can check the 'add to known faces' checkbox, type the name of the person, type a description about the person, and add him/ her in the database.

Limitations: The model is in the demo phase but the results can be enhanced via training the model using more data. 
'''





# CONSTANTS
PATH_DATA = 'db.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]


def init_data(data_path=PATH_DATA):
    # print(data_path)
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

# convert image from opened file to np.array


def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

# convert opencv BRG to regular RGB mode


def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# convert face distance to similirity likelyhood


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

# st.write("بالتجربة على الصور وجدت أنه لا يدرك بعض الوجوه الصغيرة أو غير واضحة الملامح")
if __name__ == "__main__":
    # disable warning signs:
    # https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
    st.set_option("deprecation.showfileUploaderEncoding", False)



    # displays a file uploader widget and return to BytesIO
    image_byte = st.file_uploader(
        label="Select a picture contains faces:", type=['jpg', 'png']
    )
    # detect faces in the loaded image
    max_faces = 0
    rois = []  # region of interests (arrays of face areas)
    if image_byte is not None:
        image_array = byte_to_array(image_byte)
        face_locations = face_recognition.face_locations(image_array)
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # save face region of interest to list
            rois.append(image_array[top:bottom, left:right].copy())

            # Draw a box around the face and lable it
            cv2.rectangle(image_array, (left, top),
                          (right, bottom), COLOR_DARK, 2)
            cv2.rectangle(
                image_array, (left, bottom + 35),
                (right, bottom), COLOR_DARK, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image_array, f"#{idx}", (left + 5, bottom + 25),
                font, .55, COLOR_WHITE, 1
            )

        st.image(BGR_to_RGB(image_array), width=720)
        max_faces = len(face_locations)

    if max_faces > 0:
        # select interested face in picture
        row2_1, row2_2 = st.columns((4, 6))


        with row2_1:
            # st.image('logo.png')
            face_idx =st.selectbox("Select face#", range(max_faces))


        # face_idx = st.selectbox("Select face#", range(max_faces))
        roi = rois[face_idx]
        print(roi.shape[0],"__________________________________________________")
        st.image(BGR_to_RGB(roi), width=min(5*(roi.shape[0]), 1500))

        # initial database for known faces
        DB = init_data()
        face_encodings = DB[COLS_ENCODE].values
        dataframe = DB[COLS_INFO]

        # compare roi to known faces, show distances and similarities
        # print(type(face_recognition.face_encodings(roi)))
        print(len(face_recognition.face_encodings(roi)), "::::::::::::::::::::::::::::::::::")
        print(face_recognition.face_encodings(roi)[0], "##################")
        face_to_compare = face_recognition.face_encodings(roi)[0]# ac solution to the error of 'list index out of range' : https://github.com/ageitgey/face_recognition/wiki/Common-Errors#issue-assertion-failed-ssizewidth--0--ssizeheight--0-when-running-the-webcam-examples
        print(face_to_compare,"++++++++++++++++++++++++++++++++++++++")
        dataframe['distance'] = face_recognition.face_distance(
            face_encodings, face_to_compare
        )
        dataframe['similarity'] = dataframe.distance.apply(
            lambda distance: f"{face_distance_to_conf(distance):0.2%}"
        )
        st.dataframe(
            dataframe.sort_values("distance").iloc[:10]
            .set_index('name')
        )

        # add roi to known database
        if st.checkbox('add it to knonwn faces'):
            face_name = st.text_input('Name:', '')
            face_des = st.text_input('Desciption:', '')
            if st.button('add'):
                encoding = face_to_compare.tolist()
                DB.loc[len(DB)] = [face_name, face_des] + encoding
                DB.to_csv(PATH_DATA, index=False)
    else:
        st.write('No human face detected.')

    st.info(
        "The application can be integrated with any solution that requires determining the identity or the presence of a person")
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
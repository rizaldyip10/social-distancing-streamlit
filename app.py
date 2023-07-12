# Import library
import streamlit as st
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import datetime
import wget
from PIL import Image
import pandas as pd
import csv
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
import plotly.figure_factory as ff

# Database
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB functions for video
def create_video_table():
    c.execute('CREATE TABLE IF NOT EXISTS videotable(author TEXT,title TEXT,file_date DATE,path TEXT)')

def add_video(author,title,file_date,path):
    c.execute('INSERT INTO videotable(author,title,file_date,path) VALUES (?,?,?,?)', (author,title,file_date,path))
    conn.commit()

def view_all_videos():
    c.execute("SELECT * FROM videotable")
    data = c.fetchall()
    return data

def view_by_video_author():
    c.execute('SELECT DISTINCT author FROM videotable')
    data = c.fetchall()
    return data

def get_path_by_video_author(author):
    c.execute('SELECT path FROM videotable WHERE author="{}"'.format(author))
    data = c.fetchall()
    return data

# DB Functions for Image
def create_image_table():
    c.execute('CREATE TABLE IF NOT EXISTS imagetable(author TEXT,title TEXT,file_date DATE,path TEXT)')

def add_image(author,title,file_date,path):
    c.execute('INSERT INTO imagetable(author,title,file_date,path) VALUES (?,?,?,?)', (author,title,file_date,path))
    conn.commit()

def view_all_images():
    c.execute("SELECT * FROM imagetable")
    data = c.fetchall()
    return data

def view_by_image_author():
    c.execute('SELECT DISTINCT author FROM imagetable')
    data = c.fetchall()
    return data

def get_path_by_image_author(author):
    c.execute('SELECT path FROM imagetable WHERE author="{}"'.format(author))
    data = c.fetchall()
    return data

def delete_image(path):
    c.execute('DELETE FROM imagetable WHERE path="{}"'.format(path))
    conn.commit()

def delete_video(path):
    c.execute('DELETE FROM videotable WHERE path="{}"'.format(path))
    conn.commit()


# save & upload helper function
def save_uploaded_image(uploaded_image):
    with open(os.path.join("images",uploaded_image.name),"wb") as f:
        f.write(uploaded_image.getbuffer())

def save_uploaded_video(uploaded_video):
    with open(os.path.join("videos",uploaded_video.name),"wb") as f:
        f.write(uploaded_video.getbuffer())


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    f = open("data.csv", "w")
    f.truncate()
    f.close()

    sidebar = st.sidebar.selectbox('Choose input to analyse', ('Video','Webcam/CCTV', 'View file uploaded'))

    if sidebar == 'Video':
        video_analysis()
    if sidebar == "Webcam":
        webcam()
    if sidebar == 'View file uploaded':
        view_file()

def video_analysis():
    st.title('Video Analysis for Social Distancing Monitoring')
    st.subheader("Upload Your Image")
    create_video_table()
    file_author = st.text_input("Enter your name", max_chars=50)
    file_title = st.text_input("Enter Desire File Name")
    file_date = st.date_input("Created Date")
    video_file = st.file_uploader("Upload An Image",type=['mp4'])

    if video_file is not None:
        file_details = {"FileName":video_file.name,"FileType":video_file.type}
        img = video_file.read()
        st.video(img)
        path = os.path.join("videos", video_file.name)
        save_uploaded_video(video_file)

    if st.button("Add"):
        add_video(file_author, file_title, file_date, path)
        st.success("File: {} saved".format(file_title))

    st.subheader("Social Distancing Monitoring System with Images")
    cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

    st.subheader('Test Demo Video')
    all_titles = [i[0] for i in view_by_video_author()]
    option = st.selectbox('Your Name', all_titles)
    all_path = [i[0] for i in get_path_by_video_author(option)]
    option2 = st.selectbox("Select your uploaded file", all_path)


    USE_GPU = bool(cuda)
    MIN_DISTANCE = 50


    labelsPath = "model/obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = "model/yolov4-obj_final.weights"
    configPath = "model/yolov4-obj.cfg"

    # Create a header for CSV file for category
    header = ['safe_count','violate_count', "violate_limit"]


    #CSV open and amend CSV file
    with open('data.csv', 'a') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(header)
        f.close()

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    if USE_GPU:
        st.info("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    
    st.write("The graph wil auto generate with a time interval of 2 seconds")


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if st.button('Start'):
        start = time.time()

        st.info("[INFO] loading YOLO from disk...")
        st.info("[INFO] accessing video stream...")

        for i in view_by_video_author():
            if option == i[0]:
                vs = cv2.VideoCapture(option2)
            else:
                vs = cv2.VideoCapture(0)
            writer = None
            image_placeholder = st.empty()

        while True:

            (grabbed, frame) = vs.read()

            if not grabbed:
                break

            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                                    personIdx=LABELS.index("Person"))

            violate = set()
            total = set()

            if len(results) >= 2:

                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):

                        if D[i, j] < MIN_DISTANCE:

                            violate.add(i)
                            violate.add(j)

                        elif D[i, j] >= MIN_DISTANCE:

                            total.add(i)
                            total.add(j)

                        safe = total - violate

            for (i, (prob, bbox, centroid)) in enumerate(results):

                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                if i in violate:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            datet = str(datetime.datetime.now())
            frame = cv2.putText(frame, datet, (0, 35), font, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)
            text = "Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            text_safe = "Safe: {}".format(len(safe))
            cv2.putText(frame, text_safe, (250, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (17, 59, 8), 3)


            count_violate = str(len(violate))
            count_safe = str(len(safe))
            count_datet = str(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'))



            count_violate_list = []
            count_safe_list = []
            count_datet_list = []
            max_violate = []

            count_violate_list.append(count_violate)
            count_safe_list.append(count_safe)

            if count_violate_list == [] and count_safe_list == []:
                max_violate.append(0)
            else:
                max_violate.append((int(count_safe_list[-1]) + int(count_violate_list[-1])) // 2)

            # end = (time.time() / (24 * 3600 * 1000))
            # difference = (end - start)            
            count_datet_list.append(count_datet)

            with open('data.csv', 'a') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerows(zip(count_datet_list,count_safe_list, count_violate_list, max_violate))
                    f.close()     
            
            if count_violate > count_safe:
               warning = st.warning('Room is too crowded!', icon="⚠️")
            else:
                warning = st.warning('')
            

            col1, col2, col3 = st.columns(3)

            limit_metric = col1.metric("**Violators limit**", max_violate[-1])
            violation_metric = col2.metric("**:red[Total violators]**", count_violate_list[-1])
            safe_metric = col3.metric("**:green[At safe distance]**", count_safe_list[-1])
            

            display = 1
            if display > 0:

                image_placeholder.image(
                    frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

            if writer is not None:
                writer.write(frame)

            DATA_URL=('data.csv')
            @st.cache(persist=True)
            def load_data():
                data=pd.read_csv(DATA_URL)
                return data

            df = pd.read_csv(DATA_URL)

            df.to_csv('Data_Results.csv')

            out = cv2.VideoWriter("./Demo/test_output_4.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))

            chart_caption = st.text('Chart of Number of Person Violated vs Person in Safety Disatnce')
            
            ###################################################
            #chart_caption = st.text('Line chart of violated quantity & time in (s)')
            linechart = st.line_chart(df)

            countdown = st.text('Countdown: 3')
            time.sleep(1)
            countdown.empty()
            countdown = st.text('Countdown: 2')
            time.sleep(1)
            countdown.empty()
            countdown = st.text('Countdown: 1')
            time.sleep(1)
            countdown.empty()
            linechart.empty()
            chart_caption.empty()
            warning.empty()
            limit_metric.empty()
            violation_metric.empty()
            safe_metric.empty()
            


    st.success("Success!!")

def webcam():
    st.title('Real Time Video Analysis for Social Distancing Monitoring')

    cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

    USE_GPU = bool(cuda)
    MIN_DISTANCE = 50


    labelsPath = "model/obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = "model/yolov4-obj_final.weights"
    configPath = "model/yolov4-obj.cfg"

    # Create a header for CSV file for category
    header = ['violate_count','safe_count']


    #CSV open and amend CSV file
    with open('data.csv', 'a') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(header)
        f.close()

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    if USE_GPU:
        st.info("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    
    st.write("The graph wil auto generate with a time interval of 2 seconds")


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if st.button('Start'):
        start = time.time()

        st.info("[INFO] loading YOLO from disk...")
        st.info("[INFO] accessing video stream...")

        vs = cv2.VideoCapture(0)
        writer = None
        image_placeholder = st.empty()

        while True:

            (grabbed, frame) = vs.read()

            if not grabbed:
                break

            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                                    personIdx=LABELS.index("Person"))

            violate = set()
            total = set()

            if len(results) >= 2:

                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):

                        if D[i, j] < MIN_DISTANCE:

                            violate.add(i)
                            violate.add(j)

                        elif D[i, j] >= MIN_DISTANCE:

                            total.add(i)
                            total.add(j)

                        safe = total - violate

            for (i, (prob, bbox, centroid)) in enumerate(results):

                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                if i in violate:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            datet = str(datetime.datetime.now())
            frame = cv2.putText(frame, datet, (0, 35), font, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)
            text = "Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            text_safe = "Safe: {}".format(len(safe))
            cv2.putText(frame, text_safe, (250, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (17, 59, 8), 3)


            count_violate = str(len(violate))
            count_safe = str(len(safe))
            #count_datet = str(datetime.datetime.now().strftime('%M:%S.%f')[:-4])



            count_violate_list = []
            count_safe_list = []
            #count_datet_list = []

            count_violate_list.append(count_violate)
            count_safe_list.append(count_safe)

            end = time.time()
            difference = (end - start)            
            #count_datet_list.append(difference)            

            with open('data.csv', 'a') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerows(zip(count_violate_list,count_safe_list))
                    f.close()     
            
            if count_violate > count_safe:
               warning = st.warning('Social Distancing Violators Exceed Safe Number', icon="⚠️")
            else:
                warning = False
            
            display = 1
            if display > 0:

                image_placeholder.image(
                    frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

            if writer is not None:
                writer.write(frame)

            DATA_URL=('data.csv')
            @st.cache(persist=True)
            def load_data():
                data=pd.read_csv(DATA_URL)
                return data

            df = pd.read_csv(DATA_URL)

            chart_caption = st.text('Chart of Number of Person Violated vs Person in Safety Disatnce')
            
            
            ###################################################
            linechart = st.line_chart(df)
            #chart_caption = st.text('Line chart of violated quantity & time in (s)')
            #fig = go.Figure(go.Pie(df))
            #piechart = st.line_chart(fig)

            #countdown = st.text('Countdown: 3')
            #time.sleep(1)
            #countdown.empty()
            countdown = st.text('Countdown: 2')
            time.sleep(1)
            countdown.empty()
            countdown = st.text('Countdown: 1')
            time.sleep(1)
            countdown.empty()
            linechart.empty()
            chart_caption.empty()
            warning.empty()




def view_file():
    st.header("View All Files")
    images = view_all_images()
    
    image_db = pd.DataFrame(images, columns=["Author", "Title","Created Date","File Path"])
    st.subheader("Image Database")
    st.dataframe(image_db)
    all_images = [i[0] for i in view_by_image_author()]
    image_option_1 = st.selectbox('Your Name for Image', all_images)
    all_path = [i[0] for i in get_path_by_image_author(image_option_1)]
    image_option_2 = st.selectbox("Select your uploaded image", all_path)
    if st.button("Delete Image"):
        delete_image(image_option_2)
        st.warning("Deleted: '{}'".format(image_option_2))
    
    videos = view_all_videos()
    st.subheader("Video Database")
    video_db = pd.DataFrame(videos, columns=["Author", "Title","Created Date","File Path"])
    st.dataframe(video_db)
    all_videos = [i[0] for i in view_by_video_author()]
    video_option_1 = st.selectbox('Your Name for Video', all_videos)
    all_video_path = [i[0] for i in get_path_by_video_author(video_option_1)]
    video_option_2 = st.selectbox("Select your uploaded video", all_video_path)
    if st.button("Delete Video"):
        delete_video(video_option_2)
        st.warning("Deleted: '{}'".format(video_option_2))




if __name__ == "__main__":
    main()
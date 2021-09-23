import streamlit as st
import cv2
import keras_ocr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import subprocess
os.system('cmd /k "date"')

PAGE_CONFIG = {"page_title": "StColab.io",
"page_icon": ":smiley:", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)


def main():
    st.title("Awesome Streamlit for ML")
    st.subheader("How to run streamlit from colab")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
            st.subheader("Streamlit From Colab")

    img_file_buffer = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
    # st.image(image, caption=f"Test", use_column_width=True,)
    cv2.imwrite('images/test.jpg', image)

    next_step = False
    if st.button('RUN'):
        with st.spinner('Loading yolo and detecting license plate....'):
            cmd = 'cd yolov5 ; python detect.py --weights weights/best-fp16.tflite --source images/test.jpg --img 640 --save-txt'
            returned = subprocess.call(cmd, shell=True)
        st.write('DONEDONEDONE')
        time.sleep(20)
        next_step = True

    if next_step == True:

            with open('runs/detect/exp/labels/test.txt', 'r') as f:
                    text = f.readlines()
            bb = []
            for line in text:
                    s = line.split()
                    x1 = int(s[1])
                    y1 = int(s[2])
                    x2 = int(s[3])
                    y2 = int(s[4])
                    bb.append((x1, y1, x2, y2))

    img = cv2.imread('images/test.jpg')
    cropped_images = []
    i = 0
    for box in bb:
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        scale_per = 3
        width = int(crop.shape[1] * scale_per)
        height = int(crop.shape[0] * scale_per)
        dim = (width, height)
        scaled = cv2.resize(crop, dim, interpolation=cv2.INTER_CUBIC)
        des = cv2.fastNlMeansDenoisingColored(scaled, None, 10, 10, 7, 15)
        cropped_images.append(des)
        st.image(des, caption=f"{x1}, {y1}, {x2}, {y2} are the bbox", use_column_width='never')
        cv2.imwrite(f'runs/detect/exp/crp{i}.jpg', crop)
        i+=1
    
    
    

    pipeline = keras_ocr.pipeline.Pipeline()
    # images = [ keras_ocr.tools.read('/content/yolov5/runs/detect/exp/test.jpg')]
    images = cropped_images
    
     # Plot the predictions
    if len(images) > 1:
        prediction_groups = pipeline.recognize(images)             
        fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
        for ax, image, predictions in zip(axs, images, prediction_groups):
            keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
        fig.savefig('result/res.jpg')
        res = cv2.imread('result/res.jpg')
        st.image(res, caption='images of lp numbers', use_column_width='never')

    else:
        images = [des]
        prediction_groups = pipeline.recognize(images)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=ax)
        fig.savefig('result/res.jpg')
        res = cv2.imread('result/res.jpg')
        lp_number = [text for text, box in prediction_groups[0]]
        st.image(res, caption=f'{lp_number}', use_column_width='never')
    delete = subprocess.call('rm -r runs', shell=True)



# fig = plt.figure()
# ax = fig.add_subplot(111)
# keras_ocr.tools.drawAnnotations(image=images[0], predictions=predictions[0], ax=ax)
    # fig.savefig('/content/yolov5/result/res.jpg')
    # res = cv2.imread('/content/yolov5/result/res.jpg')
    # st.image(res, caption='Predicted License Plate Number', use_column_width=True)
# delete = subprocess.call('rm -r /content/yolov5/runs', shell=True)

    







if __name__ == '__main__':
    main()

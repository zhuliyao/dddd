# coding:utf-8
import numpy as np
import tensorflow as tf
import cv2 as cv

# Load TFLite model and allocate tensors.
tflite_model = tf.contrib.lite.Interpreter(model_path="/home/hiicy/Downloads/ws3_vgg.tflite")
tflite_model.allocate_tensors()

# Get input and output tensors.
input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
image = cv.imread("/home/hiicy/redldw/ldw/image2_1.png")

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # 输入随机数
print('input_data',input_data.shape)
input_data[0]=image
tflite_model.set_tensor(input_details[0]['index'], input_data)

tflite_model.invoke()
output_data = tflite_model.get_tensor(output_details[0]['index'])
print("out_class")
print(output_data)
# -*- coding: utf-8 -*-
"""alexnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PRHVfMpHfIaQ_95xGFWMrTEVcVq4-UUN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

from google.colab import files
uploaded = files.upload()

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  #AlexNet architecture
  # Input Layer
  input_layer =features["x"]
    
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
   
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  #pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
  
  #Pooling layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  #pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
   
  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
   
  # Convolutional Layer #5
  conv5 = tf.layers.conv2d(inputs=conv4,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
   
  #Pooling layer #3
  pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  #pool3 = tf.layers.average_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  
  layer_shape = pool3.get_shape()
  num_features = layer_shape[1:4].num_elements()
  pool3_flat = tf.reshape(pool3, [-1, num_features])
  
  # Dense Layer
  #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
  
  #drop layer
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  #output layer
  logits = tf.layers.dense(inputs=dropout, units=3)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and TEST modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  
  #create confusion matrix
#   cm=tf.confusion_matrix(labels,predictions["classes"],num_classes=3)
#   cm=tf.Print(cm,[cm],message="CM_MAT")
#   print(cm.eval())
  
  # add evaluation metrics
  eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  
  # Load training data
  datainfo_train=pd.read_csv("transformed_data_info_2_train.csv")
  instances_column=datainfo_train['ref_num']
  train_labels=datainfo_train['severity']
  train_data=[] 
  for p in range(len(instances_column)):
    imname=instances_column[p]+'.pgm'
    image=cv2.imread(imname)
    train_data.append(image)
   
  train_data = np.asarray(train_data, dtype=np.float32)
  classmap={'N':0,'B':1,'M':2}
  train_labels=train_labels.map(classmap)#create numerical labels
  
  #load test data
  datainfo_test=pd.read_csv("transformed_data_info_2_test.csv")
  instances_column=datainfo_test['ref_num']
  test_labels=datainfo_test['severity']
  test_data=[]
  
  for p in range(len(instances_column)):
    imname=instances_column[p]+'.pgm'
    image=cv2.imread(imname)
    test_data.append(image)
   
  test_data = np.asarray(test_data, dtype=np.float32)
  classmap={'N':0,'B':1,'M':2}
  test_labels=test_labels.map(classmap) #create numerical labels
   
  # Create the estimator
  cancer_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/_convnet_model")
  
 # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=490,num_epochs=None,shuffle=True)
  cancer_classifier.train(input_fn=train_input_fn,steps=50,hooks=[logging_hook])
  
  # Evaluate the model on test and train data
  eval_input_fn_test = tf.estimator.inputs.numpy_input_fn(x={"x": test_data},y=test_labels,num_epochs=1,shuffle=False)
  eval_results = cancer_classifier.evaluate(input_fn=eval_input_fn_test)
  print('TEST ACCURACY: ')
  print(eval_results)
  
  eval_input_fn_train = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,num_epochs=1,shuffle=False)
  eval_results = cancer_classifier.evaluate(input_fn=eval_input_fn_train)
  print('TRAINING ACCURACY: ')
  print(eval_results)
  
  eval_input_fn_predict = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},num_epochs=1,shuffle=False)
  pred = cancer_classifier.predict(input_fn=eval_input_fn_predict)
  train_y=[]
  print(type(pred))
  for i in pred:
    train_y.append(i["classes"])
  
  from sklearn.metrics import confusion_matrix
  cm=confusion_matrix(train_labels, train_y, labels=[0, 1, 2])
  print(cm)
  
  eval_input_test_predict = tf.estimator.inputs.numpy_input_fn(x={"x": test_data},num_epochs=1,shuffle=False)
  pred = cancer_classifier.predict(input_fn=eval_input_test_predict)
  test_y=[]
  for i in pred:
    test_y.append(i["classes"])
  cm=confusion_matrix(test_labels, test_y, labels=[0, 1, 2])
  print(cm)
  print(cm.ravel())
  cm=confusion_matrix(test_labels, test_labels, labels=[0, 1, 2])
  print(cm)
#   cm=tf.confusion_matrix(train_labels,predictions_,num_classes=3)
#   print(cm)
#   cm=tf.Print(cm,[cm],message="CM_MAT")

main(1)

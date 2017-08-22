import tensorflow as tf 
#To remove AVX,AVX2 warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
x1=tf.constant(5)
x2=tf.constant(6)

result = tf.multiply(x1,x2)

print(result)

with tf.Session() as sess:
	print(sess.run(result))
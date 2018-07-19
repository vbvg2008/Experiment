'''
This script shows how to convert the mnist data into tfrecords
'''
import sys
import os
import tensorflow as tf
import argparse

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _byptes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(floats_list = tf.train.FloatList(value = value))

def createDataRecord(out_dir, out_filename, x, y):
    out_path = os.path.join(out_dir, out_filename)
    #open the TF record writer
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i in range(x.shape[0]):
            #Output progress
            if i % 1000 == 0 and i != 0:
                print('data: {}/{}'.format(i, x.shape[0]))
                sys.stdout.flush()
            img = x[i, :, :]
            label = y[i]
            
            #Create feature dictionary
            feature = {
                    'image_raw': _byptes_feature(img.tostring()),
                    'label'    : _int64_feature(label)
                    }
            
            #Define example protocol buffer
            example = tf.train.Example(features = tf.train.Features(feature = feature))
            
            #Serialzie to string and write on the file
            writer.write(example.SerializeToString())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mnist_dir', 
                        type = str, 
                        nargs = '?',
                        default = '', 
                        help = 'The directory to save mnist data')
    
    parser.add_argument('--out_dir', 
                    type = str, 
                    nargs ='?',
                    default = '', 
                    help = 'The directory to output tfrecords')
    
    args = parser.parse_args()
    
    mnist_path = os.path.join(args.mnist_dir, 'mnist.npz')
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(mnist_path)
    
    createDataRecord(args.out_dir, 'train.tfrecords', x_train, y_train)
    
    createDataRecord(args.out_dir, 'test.tfrecords', x_test, y_test)

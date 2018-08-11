#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'  # Pool 3
    vgg_layer4_out_tensor_name = 'layer4_out:0'  # Pool 4
    vgg_layer7_out_tensor_name = 'layer7_out:0'  # Fully connected 7

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    it = graph.get_tensor_by_name(vgg_input_tensor_name)
    kpt = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3ot = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4ot = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7ot = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return it, kpt, l3ot, l4ot, l7ot
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    stddev_1x1 = 1e-3
    l2_reg_rate = 1e-2
    stddev_deconv_2x2 = 1e-2

    # Scale layer3_out (Pool3) output by 0.0001
    layer3_out_scaled = tf.multiply(vgg_layer3_out, 1e-4, name='layer3_out_scaled')

    # 1x1 convolution after layer3_out (Pool3)
    layer3_out_1x1 = tf.layers.conv2d(inputs=layer3_out_scaled,
                                  filters=num_classes,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate),
                                  name='layer3_out_1x1')

    # Scale layer4_out (Pool4) output by 0.01
    layer4_out_scaled = tf.multiply(vgg_layer4_out, 1e-2, name='layer4_out_scaled')

    # 1x1 convolution after layer4_out (Pool4)
    layer4_out_1x1 = tf.layers.conv2d(inputs=layer4_out_scaled,
                                      filters=num_classes,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate),
                                      name='layer4_out_1x1')

    # 1x1 convolution after layer7_out (FullyConnected4)
    layer7_out_1x1 = tf.layers.conv2d(inputs=vgg_layer7_out,
                                      filters=num_classes,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate),
                                      name='layer7_out_1x1')

    # Deconvolution 2x2 of layer7_out_1x1
    layer7_out_deconv_2x2 = tf.layers.conv2d_transpose(inputs=layer7_out_1x1,
                                              filters=num_classes,
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(
                                                  stddev=stddev_deconv_2x2),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate),
                                              name='layer7_out_deconv_2x2')

    # Add of layer7_out_deconv_2x2 and layer4_out_1x1
    skip1 = tf.add(layer7_out_deconv_2x2, layer4_out_1x1, name='skip1')

    # Deconvolution 2x2 of skip1
    skip1_deconv_2x2 = tf.layers.conv2d_transpose(inputs=skip1,
                                                   filters=num_classes,
                                                   kernel_size=(4, 4),
                                                   strides=(2, 2),
                                                   padding='same',
                                                   kernel_initializer=tf.truncated_normal_initializer(
                                                       stddev=stddev_deconv_2x2),
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate),
                                                   name='skip1_deconv_2x2')

    # Add of skip1_deconv_2x2 and layer3_out_1x1
    skip2 = tf.add(skip1_deconv_2x2, layer3_out_1x1, name='skip2')

    # Deconvolution 8x8 of skip2
    skip2_deconv_8x8 = tf.layers.conv2d_transpose(inputs=skip2,
                                                  filters=num_classes,
                                                  kernel_size=(16, 16),
                                                  strides=(8, 8),
                                                  padding='same',
                                                  kernel_initializer=tf.truncated_normal_initializer(
                                                      stddev=stddev_deconv_2x2),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate),
                                                  name='skip1_deconv_8x8')

    return skip2_deconv_8x8
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape last network layer and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Classification loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels),
        name='cross_entropy_loss'
    )

    # Regularization loss
    # List of the individual loss values
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    print("Regression loss collection: {}".format(regularization_losses))
    regularization_loss = sum(regularization_losses)

    # Total loss
    total_loss = tf.add(cross_entropy_loss, regularization_loss, name='total_loss')

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')

    # Minimize loss function
    train_op = optimizer.minimize(total_loss, name='train_op')

    return logits, train_op, total_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Init global variables
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):

        loss_evolution = []
        epoch_cost = 0
        batch_counter = 0

        for image, label in get_batches_fn(batch_size):
            # Training
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={
                    input_image: image,
                    correct_label: label,
                    keep_prob: 0.5,
                    learning_rate: 0.00001
                }
            )
            loss_evolution.append("{:4f}".format(loss))
            epoch_cost = epoch_cost + loss
            batch_counter = batch_counter + 1

        epoch_cost = epoch_cost / batch_counter
        print("Epoch {} loss: {}".format(e + 1, epoch_cost))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    models_dir = './models'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # NN Training
        epochs = 100  # 100 50 20 10
        batch_size = 15  # 15 10 5

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # Save inference data and model
        helper.save_inference_samples(runs_dir, models_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, tf.train.Saver())


if __name__ == '__main__':
    run()

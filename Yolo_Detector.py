import tensorflow as tf
_ANCHORS = []
_MODEL_SIZE = ()

# 2-D convolution
def Conv2d(inputs, filters, kernel_size, strides = 1):
    return tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = ('SAME' if strides == 1 else 'VALID'))

# Batch normalization
def BatchNorm(inputs):
    return tf.layers.batch_normalization(inputs = inputs, axis = 1)

# Convolution operations layer block
def Convolution_block(inputs, filters):
    inputs = Conv2d(inputs, filters = filters, kernel_size = 1)
    inputs = BatchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs)

    inputs = Conv2d(inputs, filters = 2 * filters, kernel_size = 3)
    inputs = BatchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs)

    inputs = Conv2d(inputs, filters = filters, kernel_size = 1)
    inputs = BatchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs)

    inputs = Conv2d(inputs, filters = 2 * filters, kernel_size = 3)
    inputs = BatchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs)

    inputs = Conv2d(inputs, filters = filters, kernel_size = 1)
    inputs = BatchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs)
    UpOutput = inputs

    inputs = Conv2d(inputs, filters=2 * filters, kernel_size=3)
    inputs = BatchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs)
    return UpOutput, inputs

# Final detection layer of three detectors
def Detect(inputs, n_classes, anchors, img_size):
    n_anchors = len(anchors)
    inputs = tf.layers.conv2d(inputs, filters = n_anchors * (5 + n_classes), kernel_size = 1, strides = 1)
    
    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4]
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    box_centers, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis = -1)

    x = tf.range(grid_shape[0], dtype = tf.float32)
    y = tf.range(grid_shape[1], dtype = tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset, y_offset = tf.reshape(x_offset, (-1, 1)), tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis = -1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

    confidence = tf.nn.sigmoid(confidence)
    classes = tf.nn.sigmoid(classes)
    inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis = -1)
    return inputs

# Upsample the feature map using nearest neighbor interpolation
def UpSample(inputs, out_shape):
    new_height = out_shape[0]
    new_width = out_shape[1]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
    return inputs

# Main driver function for the entire feature map fusion and prediction
def Main_pipeline(F1, F2, F3, n_classes, model_size):
    F1, F1_predict = Convolution_block(F1, filters = 64)
    Detector1 = Detect(F1_predict, n_classes = n_classes, anchors = _ANCHORS[0:3], img_size = model_size)
    F1 = Conv2d(F1, filters = 128, kernel_size = 1)
    F1 = BatchNorm(F1)
    F1 = tf.nn.leaky_relu(F1)
    Up_size = F2.get_shape().as_list()
    F1 = UpSample(F1, out_shape = Up_size)

    F2 = tf.concat([F1, F2])
    F2, F2_predict = Convolution_block(F2, filters = 128)
    Detector2 = Detect(F2_predict, n_classes = n_classes, anchors = _ANCHORS[3:6], img_size = model_size)
    F2 = Conv2d(F2, filters = 256, kernel_size = 1)
    F2 = BatchNorm(F2)
    F2 = tf.nn.leaky_relu(F2)
    Up_size = F3.get_shape().as_list()
    F2 = UpSample(F2, out_shape = Up_size)

    F3, F3_predict = tf.concat([F2, F3])
    F3 = Convolution_block(F3_predict, filters = 256)
    Detector3 = Detect(F2, n_classes = n_classes, anchors = _ANCHORS[6:9], img_size = model_size)

    Detector = tf.concat([Detector1, Detector2, Detector3], axis = 1)
    return Detector
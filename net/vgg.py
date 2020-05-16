from keras import backend as K
from keras.layers import Input, Conv2D, MaxPool2D, TimeDistributed, Flatten, Dense
from net.RoiPooling import RoiPooling

def get_weight_path():
    if K.image_data_format() == 'channels_first':
        print("theano backend not avaliable for vgg net")
        return
    elif K.image_data_format() == "channels_last":
        return "models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16
    
    return get_output_length(width), get_output_length(height)

def nn_base(input_tensor=None, trainable=False):
    if K.image_data_format() == "channels_first":
        input_shape = (3, None, None)
    elif K.image_data_format() == "channels_last":
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block1
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)

    return x

def rpn(base_layers, num_anchors):
    x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="normal", name="rpn_conv1")(base_layers)
    
    x_cls = Conv2D(num_anchors, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="rpn_out_class")(x)

    x_reg = Conv2D(num_anchors * 4, (1, 1), activation="linear", kernel_initializer="zero", name="rpn_out_regress")(x)

    return [x_cls, x_reg, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    if K.backend() == "tensorflow":
        pooling_region = 7
        input_shape = (num_rois, 7, 7, 512)
    else:
        pooling_region = 7
        input_shape = (num_rois, 512, 7, 7)

    out_roi_pool = RoiPooling(pooling_region, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name="flatten"))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation="relu", name="fc1"))(out)
    out = TimeDistributed(Dense(4096, activation="relu", name="fc2"))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation="softmax", kernel_initializer="zero"), name="dense_class_{}".format(nb_classes))(out)
    out_reg = TimeDistributed(Dense(4 * (nb_classes - 1), activation="linear", kernel_initializer="zero"), name="dense_regress_{}".format(nb_classes))(out)

    return [out_class, out_reg]
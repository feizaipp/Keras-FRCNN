from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == "tensorflow":
    import tensorflow as tf

class RoiPooling(Layer):

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {"channels_first", "channels_last"}, "dim_ordering must be in {channels_first, channels_last}"
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == "channels_first":
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == "channels_last":
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == "channels_first":
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        elif self.dim_ordering == "channels_last":
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)
            
            num_pool_regions = self.pool_size

            if self.dim_ordering == "channels_first":
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        x2 = x1 + K.maximum(1,x2-x1)
                        y2 = y1 + K.maximum(1,y2-y1)

                        new_shape = [input_shape[0], input_shape[1], y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)
            elif self.dim_ordering == "channels_last":
                x = K.cast(x, "int32")
                y = K.cast(y, "int32")
                h = K.cast(h, "int32")
                w = K.cast(w, "int32")
                rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
                outputs.append(rs)

        final_outputs = K.concatenate(outputs, axis=0)
        final_outputs = K.reshape(final_outputs, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        if self.dim_ordering == "channels_first":
            final_outputs = K.permute_dimensions(final_outputs, (0, 1, 4, 2, 3))
        elif self.dim_ordering == "channels_last":
            final_outputs = K.permute_dimensions(final_outputs, (0, 1, 2, 3, 4))

        return final_outputs
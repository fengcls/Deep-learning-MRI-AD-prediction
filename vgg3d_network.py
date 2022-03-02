def get_network(input_shape = [None,None,None], num_output=1, block_num = 5, first_channel_num = 4, kernel_regularizer_conv = 0.0, kernel_regularizer_fc = 1.0,
                adam_lr = 0.0001, trainable_arg = True, gap_flag = False, reg_flag = False, loss_metric='mae'):
    from keras.models import Model
    from keras.layers import Input, Convolution3D, MaxPooling3D, GlobalAveragePooling3D, Flatten, Dense, Dropout, Activation, Reshape, Lambda
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam, SGD
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras import backend as K
    from keras import metrics
    from keras.regularizers import l2, l1
    from keras.engine.topology import Layer
    import tensorflow as tf

    if K.image_dim_ordering() is 'th':
        bn_axis = 1
        inputs = Input(shape=(1, input_shape[0], input_shape[1], input_shape[2]))
    elif K.image_dim_ordering() is 'tf':
        bn_axis = -1
        inputs = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1))

    #################################################################################################################
    conv1 = Convolution3D(first_channel_num, (3, 3, 3), activation='relu', padding = 'same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(inputs)
    conv1 = Convolution3D(first_channel_num, (3, 3, 3), activation=None, padding = 'same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(conv1)
    bn1 = BatchNormalization(axis = bn_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                       beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(conv1)
    relu1 = Activation('relu')(bn1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu1)

    #################################################################################################################
    conv2 = Convolution3D(first_channel_num*2, (3, 3, 3), activation='relu', padding = 'same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(pool1)
    conv2 = Convolution3D(first_channel_num*2, (3, 3, 3), activation=None, padding = 'same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(conv2)
    bn2 = BatchNormalization(axis = bn_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv2)
    relu2 = Activation('relu')(bn2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu2)

    #################################################################################################################
    conv3 = Convolution3D(first_channel_num*4, (3, 3, 3), activation='relu', padding = 'same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(pool2)
    conv3 = Convolution3D(first_channel_num * 4, (3, 3, 3), activation=None, padding='same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(conv3)
    bn3 = BatchNormalization(axis = bn_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv3)
    relu3 = Activation('relu')(bn3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu3)

    #################################################################################################################
    conv4 = Convolution3D(first_channel_num*8, (3, 3, 3), activation='relu', padding = 'same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(pool3)
    conv4 = Convolution3D(first_channel_num * 8, (3, 3, 3), activation=None, padding='same', trainable=trainable_arg,
                          kernel_regularizer=l2(kernel_regularizer_conv))(conv4)
    bn4 = BatchNormalization(axis = bn_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv4)
    relu4 = Activation('relu')(bn4)
    if gap_flag & (block_num == 4):
        flatten1 = GlobalAveragePooling3D()(relu4)
    else:
        pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu4)

    if block_num >= 5:
    #################################################################################################################
        conv5 = Convolution3D(first_channel_num * 16, (3, 3, 3), activation='relu', padding='same', trainable=trainable_arg,
                              kernel_regularizer=l2(kernel_regularizer_conv))(pool4)
        conv5 = Convolution3D(first_channel_num * 16, (3, 3, 3), activation=None, padding='same', trainable=trainable_arg,
                              kernel_regularizer=l2(kernel_regularizer_conv))(conv5)
        bn5 = BatchNormalization(axis = bn_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                 gamma_initializer='ones', moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones',
                                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                 gamma_constraint=None)(conv5)
        relu5 = Activation('relu')(bn5)
        if gap_flag:
            flatten1 = GlobalAveragePooling3D()(relu5)
        else:
            pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu5)

    if block_num == 6:
        #################################################################################################################
        conv6 = Convolution3D(first_channel_num * 32, (3, 3, 3), activation='relu', padding='same', trainable=trainable_arg,
                              kernel_regularizer=l2(kernel_regularizer_conv))(pool5)
        conv6 = Convolution3D(first_channel_num * 32, (3, 3, 3), activation=None, padding='same', trainable=trainable_arg,
                              kernel_regularizer=l2(kernel_regularizer_conv))(conv6)
        bn6 = BatchNormalization(axis = bn_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                 gamma_initializer='ones', moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones',
                                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                 gamma_constraint=None)(conv6)
        relu6 = Activation('relu')(bn6)
        if gap_flag:
            flatten1 = GlobalAveragePooling3D()(relu6)
        else:
            pool6 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu6)

    #################################################################################################################
    if not gap_flag:
        if block_num == 5:
            flatten1 = Flatten()(pool5)
        elif block_num == 4:
            flatten1 = Flatten()(pool4)
        elif block_num == 6:
            flatten1 = Flatten()(pool6)

    if not reg_flag:
        if num_output==1:
            fc1 = Dense(num_output, activation='sigmoid', kernel_regularizer=l2(kernel_regularizer_fc),use_bias=True)(flatten1)
        else:
            fc1 = Dense(num_output, activation='softmax', kernel_regularizer=l2(kernel_regularizer_fc), use_bias=True)(flatten1)
    else:
        fc1 = Dense(num_output, activation=None,kernel_regularizer=l2(kernel_regularizer_fc))(flatten1)

    if gap_flag:
        model = Model(output=fc1, input=inputs)
    else:
        model = Model(output=fc1,input=inputs)

    print adam_lr
    return model

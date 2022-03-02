import cPickle as pickle
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

np.random.seed(0)
tf.set_random_seed(0)

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from vgg3d_network import get_network

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#####################################################################################################################


def ad_adniall_func(
    inference_mode=False,
    roi="whole_brain",
    input_shape=[160, 192, 160],
    num_output=1,
    finetune_flag=False,
    coreg_str="affine",
    load_pretrain_model_flag=True,
    block_num=5,
    first_channel_num=4,
    kernel_regularizer_conv=0.0,
    kernel_regularizer_fc=1.0,
    adam_lr=0.0001,
    batch_size=5,
    pretrained_model_name=None,
    gap_flag=False,
    model_suffix="",
):

    if finetune_flag:
        load_pretrain_model_flag = True
        trainable_arg = False
    else:
        trainable_arg = True

    if load_pretrain_model_flag & (pretrained_model_name is None):
        print("pretrained model not provided")
        return

    if coreg_str is "affine":
        proj_dir = (
            "<NIFTI_DATA_DIR>"  # this needs to be specified to the local directory
        )
        df_data = pd.read_csv("adni_subjects.csv")

    # nifti image column
    ids = df_data["file"].as_matrix()
    # diagnosis
    dx = df_data["dx"].as_matrix()
    # cross validation folds of AD vs CN, -1 is MCI
    cvp = df_data["cross_validation_split_ad_cn"].as_matrix()
    # scan index
    scan = df_data["scan"].as_matrix()
    # RID
    rid = df_data["RID"].as_matrix()

    exclude_ids = ids[
        np.logical_or(
            np.logical_or((dx > 3), (dx == 2)), np.logical_or((cvp == -1), scan > 0)
        )
    ]
    exclude_ids_ind = map(lambda x: ids.tolist().index(x), exclude_ids)

    ids = np.delete(ids, exclude_ids_ind)
    dx = np.delete(dx, exclude_ids_ind)
    cvp = np.delete(cvp, exclude_ids_ind)
    scan = np.delete(scan, exclude_ids_ind)
    rid = np.delete(rid, exclude_ids_ind)

    dx = np.where(dx == 3, 1, 0)
    percentile_num = 99.0
    nb_epoch = 100
    
    # whether to use regional mask
    if roi is not "whole_brain":
        assert os.path.exists(
            "brain_region_masks/mask_5e-3_" + roi + "_all_1496.nii.gz"
        ), "no mask file"
        mask = np.squeeze(
            np.array(
                [
                    nib.load(
                        "brain_region_masks/mask_5e-3_" + roi + "_all_1496.nii.gz"
                    ).get_data()[11:171, 13:205, :160]
                ]
            )
        ).astype(np.bool_)
    else:
        mask = np.ones(input_shape).astype(np.bool_)

    ###################################################################################################################
    K.set_image_dim_ordering("th")

    if not inference_mode:
        if load_pretrain_model_flag:
            pretrained_model = load_model(pretrained_model_name)
        model = get_network(
            input_shape=input_shape,
            num_output=num_output,
            block_num=block_num,
            first_channel_num=first_channel_num,
            kernel_regularizer_conv=kernel_regularizer_conv,
            kernel_regularizer_fc=kernel_regularizer_fc,
            adam_lr=adam_lr,
            trainable_arg=trainable_arg,
            gap_flag=gap_flag,
        )
        model.summary()
        init_weights_fc = model.layers[-1].get_weights()
        early_stopper = EarlyStopping(patience=10, monitor="val_loss")
        
    ###################################################################################################################
    # loop over cross-validation split
    for ksplit in range(1, 11):
        test_ind = cvp == ksplit
        train_val_ind = cvp == (ksplit + 1) % 10
        train_train_ind = np.logical_and((cvp != ksplit), (cvp != ((ksplit + 1) % 10)))
        print(
            "#sessions:",
            np.sum(train_train_ind),
            np.sum(train_val_ind),
            np.sum(test_ind),
        )
        print(
            "#subjects:",
            len(np.unique(rid)),
            len(np.unique(rid[train_train_ind])),
            len(np.unique(rid[train_val_ind])),
            len(np.unique(rid[test_ind])),
        )
        print(
            "percentage of AD:",
            np.mean(dx[train_train_ind]),
            np.mean(dx[train_val_ind]),
            np.mean(dx[test_ind]),
        )
        train_train_ids = ids[train_train_ind].tolist()
        train_val_ids = ids[train_val_ind].tolist()
        test_ids = ids[test_ind].tolist()

        dx_train = dx[train_train_ind]
        dx_val = dx[train_val_ind]
        dx_test = dx[test_ind]
        if not inference_mode:
            if load_pretrain_model_flag:
                model.set_weights(pretrained_model.get_weights())  # pretrained model
                
            model_checkpoint = ModelCheckpoint(
                "model_ad_adniall_diag_"
                + str(ksplit)
                + "_{epoch:02d}_{val_acc:.3f}_pool"
                + str(block_num)
                + "_bs_"
                + str(batch_size)
                + "_sc_"
                + str(first_channel_num)
                + "_"
                + coreg_str
                + "_adam"
                + str(adam_lr)
                + model_suffix
                + ".hdf5",
                monitor="val_acc",
                save_best_only=True,
            )
            model_checkpoint2 = ModelCheckpoint(
                "model_ad_adniall_diag_"
                + str(ksplit)
                + "_{epoch:02d}_{val_loss:.3f}_pool"
                + str(block_num)
                + "_bs_"
                + str(batch_size)
                + "_sc_"
                + str(first_channel_num)
                + "_"
                + coreg_str
                + "_adam"
                + str(adam_lr)
                + model_suffix
                + "_loss.hdf5",
                monitor="val_loss",
                save_best_only=True,
            )
            print("load validation data:")
            img_val = np.array(
                [
                    nib.load(proj_dir + str(id)).get_data()[11:171, 13:205, :160] * mask
                    for id in tqdm(train_val_ids)
                ]
            ).astype("float32")

            for k in range(len(img_val)):
                img_val[k] = img_val[k] / np.percentile(img_val[k], percentile_num)
            img_val[img_val > 1.0] = 1.0
            img_val = np.expand_dims(img_val, axis=1)

            print("load training data:")
            img_train = np.array(
                [
                    nib.load(proj_dir + str(id)).get_data()[11:171, 13:205, :160] * mask
                    for id in tqdm(train_train_ids)
                ]
            ).astype("float32")
            print("normalizing images:")
            for k in tqdm(range(len(img_train))):
                img_train[k] = img_train[k] / np.percentile(
                    img_train[k], percentile_num
                )
            img_train[img_train > 1.0] = 1.0

            img_train = np.expand_dims(img_train, axis=1)
            
            hist = model.fit(
                img_train,
                dx_train,
                batch_size=batch_size,
                epochs=nb_epoch,
                verbose=1,
                shuffle=True,
                validation_data=[img_val, dx_val],
                callbacks=[model_checkpoint, model_checkpoint2, early_stopper],
            )

            hist_filename = (
                "model_ad_adniall_diag_pool"
                + str(block_num)
                + "_bs_"
                + str(batch_size)
                + "_sc_"
                + str(first_channel_num)
                + "_"
                + coreg_str
                + "_adam_"
                + str(adam_lr)
                + "_reguconv_"
                + str(kernel_regularizer_conv)
                + "_regufc_"
                + str(kernel_regularizer_fc)
                + model_suffix
                + "_history.pkl"
            )
            with open(hist_filename, "wb") as output:
                pickle.dump(hist.history, output, pickle.HIGHEST_PROTOCOL)
                
            ###############################################################################################################
            hist_val_acc = np.array(hist.history["val_acc"])

            load_model_name = (
                "model_ad_adniall_diag_"
                + str(ksplit)
                + "_{0:02d}_{1:.3f}".format(
                    hist_val_acc.argmax() + 1, hist_val_acc.max()
                )
                + "_pool"
                + str(block_num)
                + "_bs_"
                + str(batch_size)
                + "_sc_"
                + str(first_channel_num)
                + "_"
                + coreg_str
                + "_adam"
                + str(adam_lr)
                + model_suffix
                + ".hdf5"
            )
        else:
            load_model_name = pretrained_model_name
            
        model = load_model(load_model_name)
        print(load_model_name)
        # model.summary()
        dx_test_predict = np.zeros(len(test_ids))
        for ktest in range(len(test_ids)):
            img_test = np.array(
                nib.load(proj_dir + str(test_ids[ktest])).get_data()[
                    11:171, 13:205, :160
                ]
                * mask
            ).astype("float32")
            img_test = img_test / np.percentile(img_test, percentile_num)
            img_test[img_test > 1.0] = 1.0
            img_test = np.expand_dims(np.expand_dims(img_test, axis=0), axis=1)
            dx_test_predict[ktest] = int(model.predict(img_test) > 0.5)
        dx_predict_acc = np.float32(sum(dx_test == dx_test_predict)) / len(dx_test)
        print("test acc:" + str(dx_predict_acc))

        # break
    K.clear_session()
    
if __name__ == "__main__":
    pretrained_model_name = "model_ad_cn_adni_only_bl_ad_cn.hdf5"
    finetune_flag = False
    coreg_str = "affine"
    load_pretrain_model_flag = False
    block_num = 5
    kernel_regularizer_conv = 0.5
    kernel_regularizer_fc = 1.0
    inference_mode = False
    first_channel_num = 16
    batch_size = 5

    # roi = 'temporal_lobe'
    roi = "whole_brain"

    input_shape = [160, 192, 160]

    gap_flag = False
    if gap_flag:
        input_shape = [None, None, None]

    model_suffix = (
        "_scan0_ad_ctl_reguconv"
        + str(kernel_regularizer_conv)
        + "_fc"
        + str(kernel_regularizer_fc)
        + "_"
        + roi
    )

    if gap_flag:
        model_suffix = "_gap" + model_suffix

    for adam_lr in [2e-5]:
        ad_adniall_func(
            inference_mode=inference_mode,
            roi=roi,
            input_shape=input_shape,
            num_output=1,
            first_channel_num=first_channel_num,
            adam_lr=adam_lr,
            batch_size=batch_size,
            block_num=block_num,
            finetune_flag=finetune_flag,
            pretrained_model_name=pretrained_model_name,
            coreg_str=coreg_str,
            load_pretrain_model_flag=load_pretrain_model_flag,
            kernel_regularizer_conv=kernel_regularizer_conv,
            kernel_regularizer_fc=kernel_regularizer_fc,
            gap_flag=gap_flag,
            model_suffix=model_suffix,
        )

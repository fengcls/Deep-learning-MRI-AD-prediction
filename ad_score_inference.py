import os

import numpy as np
from keras.models import load_model
from vis.visualization import visualize_saliency, visualize_cam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import sys

import nibabel as nib
from keras import backend as K

K.set_image_dim_ordering("th")

pretrained_model_name = sys.argv[1]
model = load_model(pretrained_model_name)
# model.summary()
proj_dir = "<NIFTI_DATA_DIR>"
proj_cam_dir = "<CAM_OUTPUT_DIR>"

saliency_flag = False
cam_flag = False

ext = ".nii.gz"
ids = sorted(glob.glob(proj_dir + "*" + ext))

exclude_ids = []
exclude_ids_ind = map(lambda x: ids.tolist().index(x), exclude_ids)

ids = np.delete(ids, exclude_ids_ind)

layer_idx = -1

percentile_num = 99.0
ad_score = np.zeros((len(ids)))

for k, id in enumerate(ids):
    image_id = id.split("/")[-1].split(ext)[0]
    nib_img = nib.load(id)
    img = nib_img.get_data().astype("float32")[11:171, 13:205, :160]
    img = img / np.percentile(img, percentile_num)
    img[img > 1.0] = 1.0
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=1)
    ad_score[k] = model.predict(img)

    print(ids[k] + "," + pretrained_model_name + "," + str(ad_score[k]))
    if saliency_flag:
        for modifier in ["guided", "relu"]:
            grad_modifier = "relu"
            grads = visualize_saliency(
                model,
                layer_idx,
                filter_indices=0,
                seed_input=img,
                backprop_modifier=modifier,
                grad_modifier=grad_modifier,
            )
            img_new = np.zeros(nib_img.get_data().shape)
            img_new[11:171, 13:205, :160] = grads
            nib_img_new = nib.Nifti1Image(img_new, nib_img.affine)
            nib.save(
                nib_img_new,
                proj_cam_dir
                + image_id
                + "_visualize_saliency_test_bp"
                + modifier
                + "_grad"
                + grad_modifier
                + ext,
            )

    if cam_flag:
        for modifier in [None]:  # [None, 'guided', 'relu']:
            grad_modifier = "relu"
            grads = visualize_cam(
                model,
                layer_idx,
                filter_indices=0,
                seed_input=img,
                backprop_modifier=modifier,
                grad_modifier=grad_modifier,
            )
            img_new = np.zeros(nib_img.get_data().shape)
            img_new[11:171, 13:205, :160] = grads
            nib_img_new = nib.Nifti1Image(img_new, nib_img.affine)
            if modifier is None:
                nib.save(
                    nib_img_new,
                    proj_cam_dir
                    + image_id
                    + "_visualize_cam_test_bp"
                    + "_grad"
                    + grad_modifier
                    + ext,
                )
            else:
                nib.save(
                    nib_img_new,
                    proj_cam_dir
                    + image_id
                    + "_visualize_cam_test_bp"
                    + modifier
                    + "_grad"
                    + grad_modifier
                    + ext,
                )

f = open("output_ad_score_" + pretrained_model_name + ".txt", "w")
for k in range(len(ids)):
    f.write(ids[k] + "\t" + str(ad_score[k]) + "\n")
    print(ids[k] + "," + str(ad_score[k]))
f.close()

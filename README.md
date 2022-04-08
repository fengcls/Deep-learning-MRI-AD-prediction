# Deep learning MRI AD prediction
## Introduction
We use deep learning method for Alzheimer's disease (AD) prediction based on structural T1 MRI. This method can generate a 0-1 AD score for any brain T1 MRI image which can be used as a biomarker to evaluate the likelihood of Alzheimer's disease. In reference [1], we use the method for prodromal AD diagnosis.

There can be other similar use cases classifying different neurological status (aging ref[3], Schizophrenia, etc.) with different imaging modalities (T2 MRI, PET, fMRI, etc.), [preliminary studies](https://academiccommons.columbia.edu/doi/10.7916/d8-sshc-zj10).

## Model training and inference
* ad_score_training.py: training script with multiple hyper-parameters to tune
* ad_score_inference.py: given a directory of preprocessed nifti images (*.nii), generate the AD score and associated class-activation-map (CAM) using pre-trained model file
  ```
  python ad_score_inference.py <MODEL_FILE>
  ```
### pre-trained model files:
* model_ad_cn_adni_only_bl_ad_cn.hdf5:
  * trained with ADNI subjects who are cognitively normal (CN) or AD at *baseline*
  * used for MCI conversion (prodromal AD) classification in *ADNI* in reference [1]
* model_ad_cn_adni_all.hdf5: 
  * trained with ADNI subjects who are CN or AD at any visits
  * can be used for any data other than ADNI
    * see some open neuroimaging datasets in this [repository](https://github.com/fengcls/neuroimaging-datasets/)

## Data
Since [data use agreement](https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf) is required in order to access ADNI data, we only provide the essential and non-sensitive information in *adni_subjects.csv*. It's straightforward to join this dataset with the full dataset from ADNI after the users gaining data access.

## Preprocessing
The input brain structural MRIs are pre-processed with standard neuroimaging softwares.

Steps:
1. FreeSurfer individual brain extraction and normalization
  * Standard FreeSurfer pre-processing, *-autorecon1* is enough
  * In FreeSurfer SUBJ_DIR
  ```
   for f in *;do mri_convert $f/mri/brainmask.mgz ${OUTPUT_DIR}/${f}_mri_brainmask.nii.gz;done
  ```
2. FSL registration
  * In ${OUTPUT_DIR}
  ```
   for f in *brainmask.nii.gz;do if [ -a ${f%.nii.gz}_mni152brain_affine_tl.nii.gz ];then echo $f;else flirt -in $f -ref /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -out ${f%.nii.gz}_mni152brain_affine_tl.nii.gz -init ${f%.nii.gz}_mni152brain_affine.mat -dof 12 -applyxfm -interp trilinear;echo $f;fi;done
  ```

## Localization/Explainability
Another important piece of this series of studies is the focus on localization/explainability.
* Class activation map is one way to pinpoint the regions important to the prediction.
* Another method is to mask the input with some prior (*brain_region_masks*) and examine the performance.


## References:
0. Feng, Xinyang, Frank A. Provenzano, and Scott A. Small. ["A deep learning MRI approach outperforms other biomarkers of prodromal Alzheimer’s disease."](https://alzres.biomedcentral.com/articles/10.1186/s13195-022-00985-x) Alzheimer's Research & Therapy 14.1 (2022): 1-11.
1. Feng, Xinyang, Jie Yang, Zachary C. Lipton, Scott A. Small, Frank A. Provenzano, and Alzheimer’s Disease Neuroimaging Initiative. ["Deep learning on MRI affirms the prominence of the hippocampal formation in Alzheimer’s disease classification."](https://www.biorxiv.org/content/10.1101/456277v1.full) bioRxiv (2018): 456277. [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:cwhB9dlQ_boJ:scholar.google.com/&output=citation&scisdr=CgWWAN_lEP2g3pJgJcE:AAGBfm0AAAAAYh1mPcH9RkFFf91TIsFUtnPDlYjlxvjX&scisig=AAGBfm0AAAAAYh1mPafzeKMM788eKOc4gEyVv6lbk93R&scisf=4&ct=citation&cd=-1&hl=en)
2. Feng, Xinyang, Frank A. Provenzano, Scott A. Small, and Alzheimer’s Disease Neuroimaging Initiative. ["Detecting prodromal Alzheimer’s disease with MRI through deep learning."](https://www.biorxiv.org/content/10.1101/813899v1.full) bioRxiv (2019): 813899. [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:2-Gv23Nkm5MJ:scholar.google.com/&output=citation&scisdr=CgWWAN_lEP2g3pJgl44:AAGBfm0AAAAAYh1mj46J3dxSFq9z_u06amj90cdzGy5h&scisig=AAGBfm0AAAAAYh1mj-aQxkxAmbAe1p9I7CBXepIQqtfv&scisf=4&ct=citation&cd=-1&hl=en)
3. Feng, Xinyang, Zachary C. Lipton, Jie Yang, Scott A. Small, Frank A. Provenzano, Alzheimer’s Disease Neuroimaging Initiative, and Frontotemporal Lobar Degeneration Neuroimaging Initiative. ["Estimating brain age based on a uniform healthy population with deep learning and structural magnetic resonance imaging."](https://www.sciencedirect.com/science/article/pii/S0197458020300361?casa_token=udH_Gjqw130AAAAA:fzx5TBPKerepUw-HizomGQT8hHRVcvxFAYGZNxcuGvxM_7gguC9uNdQcMm0ZrE_9kb7yOuK_mg) Neurobiology of aging 91 (2020): 15-25. [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:l78jQgSCJh8J:scholar.google.com/&output=citation&scisdr=CgWWAN_lEP2g3pJg3ek:AAGBfm0AAAAAYh1mxeniKiKFooq5n2gOOld1j1fCjyGf&scisig=AAGBfm0AAAAAYh1mxZ5foql60_Mz1v-TUMIEXyRBQMAd&scisf=4&ct=citation&cd=-1&hl=en)

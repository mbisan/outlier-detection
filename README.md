# Outlier Exposure with Standardized Max Logits

This is the continuation of the workd done in [Synthetic outlier generation for anomaly detection in autonomous driving](https://arxiv.org/abs/2308.02184). I pretrain the semantic segmentation models on the SHIFT and StreetHazards datasets and propose using the Standardized Max Logits on the simplified detector, which uses no additional resources compared to the segmentation model, but is trained with outlier exposure to improve outlier detection.

In this case, I use the COCO dataset as the OOD dataset, compared to the original paper that uses the ADE20K dataset. I evaluate the results on the SHIFT-nopedestrian and StreetHazards datasets, as in the original paper.

---

To use the scripts create a new environment `python3 -m venv .env`, activate and install the requirements from `requirements.txt`.

To prepare the datasets, [SHIFT](https://www.vis.xyz/shift/), [StreetHazards](https://github.com/hendrycks/anomaly-seg) and [COCO2014-val](https://cocodataset.org/#home) run the corresponding `prepare_DATASET.sh` bash script.

---

To pretrain the segmentation models use the `pretrain_shift.py` and `pretrain_sh.py` scripts. The pretrained models (weights only) are available in [this link](https://drive.google.com/file/d/1mVTMDGcwyK3w9FSCDqB-aBWCD_beYDUC/view?usp=drive_link), and should be placed on the `pretrained` folder. The Outlier Injection training is done with the `ood_train.py` script, with the following command:

```
python3 ood_train --dataset DATASET_NAME --checkpoint SAVE_PATH --epochs NUM_EPOCHS \
    --beta BETA1_PARAMETER --beta2 BETA2_PARAMETER --lr LEARNING_RATE \
    --alpha ALPHA_BLENDING --horizon HORIZON_PARAMETER --blur BLURRING (--histogram)
```

The images that the script uses from the COCO dataset during training (as outliers) are previously filtered to exclude pedestrian-like and in-distribution instances.

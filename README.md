# LID onset prediction for Parkinson's disease
Binary prediction of LID in patients with Parkinson's Disease using multi-task learning
(Classification into ***wLID*** versus ***woLID*** group).

The published paper used image classification and reconstruction for multi-task learning, but the code also covers a lesion segmentation task.  
User can choose and combine tasks from classification, segmentation, and reconstruction.  
The dataset file (names Parkinson.py) is structured for all three tasks.  
If lesion masks are unavailable or segmentation is not needed, the dataset file will need to be modified.

## ✅ Requirements
The project is based on a PyTorch Docker image with specific CUDA and cuDNN versions.

- **PyTorch:** 1.12.0  
- **CUDA:** 11.3  
- **cuDNN:** 8

The Dockerfile used to build the docker image is also provided.

To install the required Python libraries:  
- ```pip install -r requirements.txt```

## 📦 Data preprocessing
- Image preprocessing: SPM12 on Matlab R2022b  
- Data preparation for cross-validation: preparation_for_cross_validation.ipynb

## 📂 Data structure
    /your_project/
    ├── Fold_0/
    |   ├── clinical_var/
    |   │   ├── Fold_0/
    |   │   │   ├── train.xlsx           # Clinical variables for training set in Fold_0
    |   │   │   └── valid.xlsx           # Clinical variables for validation set in Fold_0
    |   │   ├── Fold_1/
    |   │   ├── Fold_2/
    |   │   ├── Fold_3/
    |   │   ├── Fold_4/
    |   │   └── test/
    |   │       └── test.xlsx            # Clinical variables for the test set
    |   │
    |   ├── Fold_0/
    |   │   ├── train/
    |   │   │   ├── NM_0007_no_img.nii.gz   # Image file for woLID group
    |   │   │   ├── NM_0007_no_mask.nii.gz  # Mask file for woLID group
    |   │   │   ├── NM_0024_yes_img.nii.gz  # Image file for wLID group
    |   │   │   └── NM_0024_yes_mask.nii.gz # Mask file for wLID group
    |   │   └── valid/
    |   │       ├── NM_0014_no_img.nii.gz
    |   │       ├── NM_0014_no_mask.nii.gz
    |   │       ├── NM_0024_yes_img.nii.gz
    |   │       └── NM_0024_yes_mask.nii.gz
    |   │
    |   ├── Fold_1/
    |   │
    |   ├── Fold_2/
    |   │
    |   ├── Fold_3/
    |   │
    |   ├── Fold_4/
    |   │
    |   └── test/
    |       ├── NM_0015_no_img.nii.gz    # Image file for woLID group in test set
    |       ├── NM_0015_no_mask.nii.gz   # Mask file for woLID group in test set
    |       ├── NM_0025_yes_img.nii.gz   # Image file for wLID group in test set
    |       └── NM_0025_yes_mask.nii.gz  # Mask file for wLID group in test set
    ├── Fold_1/
    ├── Fold_2/
    ├── Fold_3/
    └── Fold_4/



## 🔒 Data availability
The data utilized in this study are not publicly accessible due to patient privacy concerns. 
Requests to access the data may be considered upon contact and IRB approval. Please contact us via email for inquiries.

- A snapshot of a sample image visualized using ITK-SNAP is provided.  
- An example excel file (.xlsx) containing clinical variables is also provided.  

## 🚀 Training
    python train.py \
    --data-folder-dir '/workspace/data_folder' \
    --model-name 'Up_SMART_Net_Dual_CLS_REC' \
    --batch-size 33 --epochs 300 --num-workers 38 --pin-mem --lr-scheduler 'poly_lr' \
    --lr 1e-3 --min-lr 1e-4 --early-stop-epoch 150 --training-stream 'Upstream' \
    --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0' --print-freq 1 \
    --output-dir '/workspace/checkpoints_ablation_study_100' \
    --output-dir-2 '/workspace/valid_results_ablation_study_100
- Please refer to example_run.py files for more details on training in the setting of 5-fold cross-validation

## 🧪 Testing
    python test_k_fold.py --data-folder-dir '/workspace/data_folder' \
    --model-name 'Up_SMART_Net_Dual_CLS_REC' \ 
    --num-workers 10 --pin-mem --training-stream 'Upstream' \
    --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0'\
    --from-pretrained-0 '/workspace/checkpoints_ablation_study_100/epoch_149_checkpoint.pth'\
    --checkpoint-total-num 1 --print-freq 1 --output-dir '/workspace/checkpoints_ablation_study_100/test'
- Please refer to example_run.py files for more details on testing in the setting of 5-fold cross-validation

## 🤖 Machine learning
- Please refer to machine_learning.ipynb

## ⭐ Paper (for citation)
- to be updated once published

## 🙏 Acknowledgement
Our code was modified based on the released codes at [SMART-Net](https://github.com/mi2rl/SMART-Net) and [PyTorch 3D UNet](https://github.com/wolny/pytorch-3dunet).

## 📧 Contact
For questions or inquiries, please contact:
- Grace Yoojin Lee - tongtongdoggy@gmail.com


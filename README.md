# LID onset prediction for Parkinson's disease
Binary prediction of LID in patients with Parkinson's Disease using multi-task learning.
The published paper used image classification and reconstruction for multi-task learning, but the code also covers a lesion segmentation task.
User can choose and combine tasks from classification, segmentation, and reconstruction.
The dataset file (names Parkinson.py) is structured for all three tasks.
If lesion masks are unavailable or segmentation is not needed, the dataset file will need to be modified.

## ✅ Requirements
pip install -r requirements.txt

## 📦 Data preprocessing

## 📂 Data structure
'''
/your_project/
├── clinical_var/
│   ├── Fold_0/
│   │   ├── train.xlsx           # Clinical variables for training set in Fold_0
│   │   └── valid.xlsx           # Clinical variables for validation set in Fold_0
│   ├── Fold_1/
│   │   ├── train.xlsx
│   │   └── valid.xlsx
│   ├── Fold_2/
│   │   ├── train.xlsx
│   │   └── valid.xlsx
│   ├── Fold_3/
│   │   ├── train.xlsx
│   │   └── valid.xlsx
│   ├── Fold_4/
│   │   ├── train.xlsx
│   │   └── valid.xlsx
│   └── test/
│       └── test.xlsx            # Clinical variables for the test set
│
├── Fold_0/
│   ├── train/
│   │   ├── NM_0007_no_img.nii.gz   # Image file for woLID group
│   │   ├── NM_0007_no_mask.nii.gz  # Mask file for woLID group
│   │   ├── NM_0024_yes_img.nii.gz  # Image file for wLID group
│   │   └── NM_0024_yes_mask.nii.gz # Mask file for wLID group
│   └── valid/
│       ├── NM_0014_no_img.nii.gz
│       ├── NM_0014_no_mask.nii.gz
│       ├── NM_0024_yes_img.nii.gz
│       └── NM_0024_yes_mask.nii.gz
│
├── Fold_1/
│   ├── train/
│   └── valid/
│
├── Fold_2/
│   ├── train/
│   └── valid/
│
├── Fold_3/
│   ├── train/
│   └── valid/
│
├── Fold_4/
│   ├── train/
│   └── valid/
│
└── test/
    ├── NM_0015_no_img.nii.gz    # Image file for woLID group in test set
    ├── NM_0015_no_mask.nii.gz   # Mask file for woLID group in test set
    ├── NM_0025_yes_img.nii.gz   # Image file for wLID group in test set
    └── NM_0025_yes_mask.nii.gz  # Mask file for wLID group in test set
'''

## 🔒 Data availability
The data utilized in this study are not publicly accessible due to patient privacy concerns. 
Requests to access the data may be considered upon contact and IRB approval. Please contact us via email for inquiries.

## 🚀 Training


## 🧪 Testing

## ⭐ Paper (for citation)
- to be updated once published

## 🙏 Acknowledgement

## 📧 Contact
For questions or inquiries, please contact:
- Grace Yoojin Lee - tongtongdoggy@gmail.com


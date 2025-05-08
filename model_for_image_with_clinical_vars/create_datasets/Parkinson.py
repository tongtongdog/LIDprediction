import re
import glob
import functools
import torch
import numpy as np
import skimage
import random
import math
import os
import random

from monai.transforms import *
from monai.data import Dataset

import pandas as pd

import warnings
warnings.filterwarnings(action='ignore') 

def list_sort_nicely(l):   
    def tryint(s):        
        try:            
            return int(s)        
        except:            
            return s
        
    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)    
    return l


def Fit_Into_Template(input, new_shape=(192,224,112)):

    image = input['image'].squeeze(0)
    mask  = input['seg_label'].squeeze(0)

    x_len, y_len, z_len = new_shape

    non_zeros = np.nonzero(image>0)
    
    new_image = np.zeros((x_len,y_len,z_len))
    new_mask = np.zeros((x_len,y_len,z_len))

    x_min = non_zeros[0].min()
    x_max = non_zeros[0].max()
    x_range = x_max - x_min
    x_start = (x_len//2) - (x_range//2)
    x_end = x_start+x_range
    
    y_min = non_zeros[1].min()
    y_max = non_zeros[1].max()
    y_range = y_max - y_min
    y_start = (y_len//2) - (y_range//2)
    y_end = y_start+y_range
    
    z_min = non_zeros[2].min()
    z_max = non_zeros[2].max()
    z_range = z_max - z_min
    z_start = (z_len//2) - (z_range//2)
    z_end = z_start+z_range
    
    new_image[x_start:x_end, y_start:y_end, z_start:z_end] = image[x_min:x_max, y_min:y_max, z_min:z_max]
    new_mask[x_start:x_end, y_start:y_end, z_start:z_end] = mask[x_min:x_max, y_min:y_max, z_min:z_max]

    input['image'] = np.expand_dims(new_image, axis=0).astype('float32')
    input['seg_label'] = np.expand_dims(new_mask, axis=0).astype('float32')

    return input


## CLAHE
def clahe_keep_depths(image, clipLimit, tileGridSize):
    image = skimage.util.img_as_ubyte(image.squeeze(0))
    
    assert image.dtype == np.uint8
    assert len(image.shape) == 3  # 2d --> (H, W, 1) / 3d --> (H, W, D)

    clahe_mat   = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    stacked_img = np.stack([clahe_mat.apply(image[..., i]) for i in range(image.shape[-1])], axis=-1)
    
    stacked_img = skimage.util.img_as_float32(stacked_img)        

    return np.expand_dims(stacked_img, axis=0)


def process_clin_vars(raw_clin_df):

    pet_list = raw_clin_df['PET_ID'].tolist()
    combined_list = []

    for index, row in raw_clin_df.iterrows():
        row_tensor = row.values[1:].astype(float)
        combined_list.append(row_tensor)

    processed_clin_df = pd.DataFrame({'PET_ID': pet_list, 'clin_array': combined_list})

    return processed_clin_df


## for corrupting the clinical variables data
def generate_marginal_distributions(df):
    """Generate empirical marginal distributions for each numerical column in a DataFrame."""
    df = df.drop(['PET_ID'], axis=1)      ## PET_ID column is for the subject ID, not a feature, so drop it
    marginal_distributions = {}
    for col in df.columns:
        marginal_distributions[col] = df[col].values
    return marginal_distributions


def corrupt(data, marginal_distributions, corruption_rate=0.3):
    """Corrupt data by replacing some of the values with random samples from their marginal distributions."""
    ## here, data is a pandas series ##
    corrupted_data = data.copy()
    num_features_to_corrupt = int(len(corrupted_data) * corruption_rate)
    features_to_corrupt = random.sample(marginal_distributions.keys(), num_features_to_corrupt)
    
    for feature in features_to_corrupt:
        if feature in marginal_distributions:
            sampled_value = np.random.choice(marginal_distributions[feature])
            corrupted_data[feature] = sampled_value
    
    return corrupted_data    ## returns a pandas series with corrupted data




## Dataset for Training #####################################################################################
def PD_Uptask_Dataset(mode, data_folder_dir="/workspace", excel_folder_dir='/workspace_clin'):
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_img.nii.gz"))
        seg_label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_mask.nii.gz"))
        
        ## no -> [1., 0.], yes -> [0., 1.]
        cls_label_list = []
        for i in range(len(img_list)):
            a_clf_label = img_list[i].split("_")[-2]
            if a_clf_label == 'yes':  cls_label_list.append(1)
            elif a_clf_label == 'no': cls_label_list.append(0)

        ## for WeightedRandomSampler() in train.py file
        cls_sample_count = np.array([len(np.where(cls_label_list==t)[0]) for t in np.unique(cls_label_list)])
        weight = 1. / cls_sample_count
        weight_vector = np.array([weight[t] for t in cls_label_list])
        weight_vector = torch.from_numpy(weight_vector)


        cls_label_list = torch.nn.functional.one_hot(torch.as_tensor(cls_label_list).to(torch.int64)).float()

        ## clinical variables added
        clinical_var_list = []
        raw_clin_df = pd.read_excel(os.path.join(excel_folder_dir,'train.xlsx'), sheet_name=0)
        marginal_distributions = generate_marginal_distributions(raw_clin_df)

        # corruption_rate = 0.5   ## you can change the corruption rate if needed
        corruption_rate = -1   ## no corruption

        for x in img_list:
            x_name = os.path.basename(x)[:-11]
            x_name = x_name.split('_')[0]+'_'+x_name.split('_')[1]

            clinical_data_to_corrupt = raw_clin_df[raw_clin_df['PET_ID'] == x_name].iloc[0,:]
            clinical_data_to_corrupt = clinical_data_to_corrupt.drop('PET_ID')

            if random.random() < corruption_rate:
                clinical_data_corrupted = corrupt(clinical_data_to_corrupt, marginal_distributions, 0.3)
            else:
                clinical_data_corrupted = clinical_data_to_corrupt

            clinical_data_corrupted_ar = clinical_data_corrupted.values.astype(np.float32)

            clinical_var_list.append(torch.tensor(clinical_data_corrupted_ar, dtype=torch.float))


        data_dicts   = [{"image": image_name, "seg_label": seg_label_name, "cls_label": cls_label_name, "clinical_var": clinical_var_name} 
                        for image_name, seg_label_name, cls_label_name, clinical_var_name in zip(img_list, seg_label_list, cls_label_list, clinical_var_list)]   

        print("Train [Total]  number = ", len(img_list))
        print("Train [LID-yes]   number = ", len([i for i in img_list if "_yes_" in i]))
        print("Train [LID-no] number = ", len([i for i in img_list if "_no_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "seg_label"]),
                AddChanneld(keys=["image", "seg_label"]),
                Orientationd(keys=["image", "seg_label"], axcodes="PLS"),

                # Pre-processing
                # Fit_Into_Template,
                # Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast
                
                # Augmentation
                RandAffined(keys=["image", "seg_label"], translate_range=(5,0,5), padding_mode='zeros', prob=0.5),              # crop 데이터 사용 시에는 translate를 너무 많이 하면 striatum이 영상 밖으로 나가버릴 수 있어 조정함
                # RandAffined(keys=["image", "seg_label"], translate_range=(10,10,10), padding_mode='zeros', prob=0.5),         # translation in x,y,z axis in range [-10,10]
                RandZoomd(keys=["image", "seg_label"], min_zoom=0.9, max_zoom=1.1, padding_mode='minimum', prob=0.5),                      # zoom in or out from 0.9 to 1.1
                RandRotated(keys=["image", "seg_label"], range_x=math.radians(15), range_y=math.radians(15), range_z=math.radians(15), prob=0.5),     # rotation by -15 to +15 degrees in all three axis
                RandFlipd(keys=["image", "seg_label"], spatial_axis=-3, prob=0.5),                                                                     # horizontal flip 
                RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.5),                                                         # random Gaussian noise
                RandGaussianSmoothd(keys=["image"], prob=0.5),                                                                           # random Gaussian smoothing
                ToTensord(keys=["image", "seg_label"])
            ]
        )   

    elif mode == 'valid':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_img.nii.gz"))
        seg_label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_mask.nii.gz"))
       
        ## no -> [1., 0.], yes -> [0., 1.]
        cls_label_list = []
        for i in range(len(img_list)):
            a_clf_label = img_list[i].split("_")[-2]
            if a_clf_label == 'yes':  cls_label_list.append(1)
            elif a_clf_label == 'no': cls_label_list.append(0)

        ## dummy variable 
        weight_vector = [] ## weighted sampler not used

        cls_label_list = torch.nn.functional.one_hot(torch.as_tensor(cls_label_list).to(torch.int64)).float()


        ## clinical variables added
        clinical_var_list = []
        raw_clin_df = pd.read_excel(os.path.join(excel_folder_dir,'valid.xlsx'), sheet_name=0)
        processed_clin_df = process_clin_vars(raw_clin_df)

        for x in img_list:
            x_name = os.path.basename(x)[:-11]
            x_name = x_name.split('_')[0]+'_'+x_name.split('_')[1]
            clinical_var_list.append(torch.tensor(processed_clin_df[processed_clin_df['PET_ID'] == x_name]['clin_array'].values[0]))


        data_dicts   = [{"image": image_name, "seg_label": seg_label_name, "cls_label": cls_label_name, "clinical_var": clinical_var_name} 
                        for image_name, seg_label_name, cls_label_name, clinical_var_name in zip(img_list, seg_label_list, cls_label_list, clinical_var_list)]   

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [LID-yes]   number = ", len([i for i in img_list if "_yes_" in i]))
        print("Valid [LID-no] number = ", len([i for i in img_list if "_no_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "seg_label"]),
                AddChanneld(keys=["image", "seg_label"]),
                Orientationd(keys=["image", "seg_label"], axcodes="PLS"),

                # Pre-processing
                # Fit_Into_Template,
                # Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast            
                
                # No Augmentation
                ToTensord(keys=["image", "seg_label"])
            ]
        )   

    return Dataset(data=data_dicts, transform=transforms), weight_vector
#################################################################################################################



## Dataset for Testing #############################################################################################
def PD_TEST_Dataset(test_dataset_name, data_folder_dir="/workspace", excel_folder_dir='/workspace_clin'):
    
    if test_dataset_name == 'Custom':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/test/*_img.nii.gz"))
        seg_label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/test/*_mask.nii.gz"))
       
        ## no -> [1., 0.], yes -> [0., 1.]
        cls_label_list = []
        for i in range(len(img_list)):
            a_clf_label = img_list[i].split("_")[-2]
            if a_clf_label == 'yes':  cls_label_list.append(1)
            elif a_clf_label == 'no': cls_label_list.append(0)
        
        cls_label_list = torch.nn.functional.one_hot(torch.as_tensor(cls_label_list).to(torch.int64)).float()
        
        ## clinical variables added
        clinical_var_list = []
        raw_clin_df = pd.read_excel(os.path.join(excel_folder_dir,'test.xlsx'), sheet_name=0)
        processed_clin_df = process_clin_vars(raw_clin_df)

        for x in img_list:
            x_name = os.path.basename(x)[:-11]
            x_name = x_name.split('_')[0]+'_'+x_name.split('_')[1]
            clinical_var_list.append(torch.tensor(processed_clin_df[processed_clin_df['PET_ID'] == x_name]['clin_array'].values[0]))


        data_dicts   = [{"image": image_name, "seg_label": seg_label_name, "cls_label": cls_label_name, "clinical_var": clinical_var_name} 
                        for image_name, seg_label_name, cls_label_name, clinical_var_name in zip(img_list, seg_label_list, cls_label_list, clinical_var_list)]   

        print("Test [Total]  number = ", len(img_list))
        print("Test [LID-yes]   number = ", len([i for i in img_list if "_yes_" in i]))
        print("Test [LID-no] number = ", len([i for i in img_list if "_no_" in i]))


        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "seg_label"]),
                AddChanneld(keys=["image", "seg_label"]),
                Orientationd(keys=["image", "seg_label"], axcodes="PLS"),

                # Pre-processing
                # Fit_Into_Template,
                # Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast    
                                
                # Normalize
                ToTensord(keys=["image", "seg_label"])
            ]
        )

    else :
        raise Exception('Error, Dataset name')        
                  
    return Dataset(data=data_dicts, transform=transforms)
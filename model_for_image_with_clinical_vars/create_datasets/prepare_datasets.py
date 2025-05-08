from create_datasets.Parkinson import *


def build_dataset(is_train, args):
    mode='train' if is_train else 'valid'
    dataset = PD_Uptask_Dataset(mode=mode, data_folder_dir=args.data_folder_dir, excel_folder_dir=args.excel_folder_dir)
    return dataset

def build_test_dataset(args):
    dataset = PD_TEST_Dataset(test_dataset_name=args.test_dataset_name, data_folder_dir=args.data_folder_dir, excel_folder_dir=args.excel_folder_dir)
    return dataset
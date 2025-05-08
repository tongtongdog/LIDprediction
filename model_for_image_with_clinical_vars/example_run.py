import os

## for training
fold_list = ['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
sub_fold_list = ['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
seed_list = [768, 187, 488, 758, 915]

## best epoch from previous training without clinical variables (with image only)
best_epoch = {'Fold_0': [133, 101, 114, 47, 129],
              'Fold_1': [142, 63, 134, 98, 78],
              'Fold_2': [149, 128, 147, 34, 45],
              'Fold_3': [91, 132, 17, 28, 45],
              'Fold_4': [109, 89, 114, 89, 2]}

for current_fold, current_seed in zip(fold_list, seed_list):

    current_best_epoch = best_epoch[current_fold]

    for idx, sub_fold in enumerate(sub_fold_list):

        os.system(f"CUDA_VISIBLE_DEVICES='0' python train.py --data-folder-dir '/workspace/{current_fold}/{sub_fold}'\
                --excel-folder-dir '/workspace/{current_fold}/clinical_var/{sub_fold}'\
                --model-name 'Up_SMART_Net_Single_CLS' --batch-size 33 --epochs 150 --warmup-epochs 0 --num-workers 10 --pin-mem --lr-scheduler 'cosine_annealing_warm_restart'\
                --lr 1e-5 --min-lr 1e-5 --early-stop-epoch 50 --training-stream 'Upstream' --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0' --print-freq 1\
                --from-pretrained-0 '/workspace/checkpoints_ablation_study_100_s{current_seed}/{sub_fold}/epoch_{current_best_epoch[idx]}_checkpoint.pth'\
                --load-weight-type 'encoder' --freeze-feature-extractor True\
                --output-dir '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/{sub_fold}'\
                --output-dir-2 '/workspace/valid_results_additional_vector_ablation_study_100_s{current_seed}/{sub_fold}'")
        


# ## for testing
fold_list = ['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
seed_list = [768, 187, 488, 758, 915]

## best epoch from training above
best_epoch = {'Fold_0': [19, 21, 13, 44, 27],
              'Fold_1': [17, 43, 13, 40, 25],
              'Fold_2': [49, 11, 42, 33, 43],
              'Fold_3': [45, 42, 14, 25, 44],
              'Fold_4': [9, 29, 19, 28, 8]}


for current_fold, current_seed in zip(fold_list, seed_list):

    current_best_epoch = best_epoch[current_fold]

    os.system(f"CUDA_VISIBLE_DEVICES='0' python test_k_fold.py --data-folder-dir '/workspace/{current_fold}'\
            --excel-folder-dir '/workspace/{current_fold}/clinical_var/test'\
            --model-name 'Up_SMART_Net_Single_CLS' --num-workers 0 --pin-mem --training-stream 'Upstream' --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0'\
            --from-pretrained-0 '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/Fold_0/epoch_{current_best_epoch[0]}_checkpoint.pth'\
            --from-pretrained-1 '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/Fold_1/epoch_{current_best_epoch[1]}_checkpoint.pth'\
            --from-pretrained-2 '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/Fold_2/epoch_{current_best_epoch[2]}_checkpoint.pth'\
            --from-pretrained-3 '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/Fold_3/epoch_{current_best_epoch[3]}_checkpoint.pth'\
            --from-pretrained-4 '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/Fold_4/epoch_{current_best_epoch[4]}_checkpoint.pth'\
            --checkpoint-total-num 5 --print-freq 1 --output-dir '/workspace/checkpoints_additional_vector_ablation_study_100_s{current_seed}/test'")
    



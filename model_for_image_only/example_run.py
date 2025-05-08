import os

## For training
fold_list = ['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
seed_list = [768, 187, 488, 758, 915]

for current_fold, current_seed in zip(fold_list, seed_list):

    for sub_fold in fold_list:

        os.system(f"CUDA_VISIBLE_DEVICES='0' python train.py --data-folder-dir '/workspace/{current_fold}/{sub_fold}'\
                --model-name 'Up_SMART_Net_Dual_CLS_REC' --batch-size 33 --epochs 300 --num-workers 38 --pin-mem --lr-scheduler 'poly_lr'\
                --lr 1e-3 --min-lr 1e-4 --early-stop-epoch 150 --training-stream 'Upstream' --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0' --print-freq 1\
                --output-dir '/workspace/checkpoints_ablation_study_100_s{current_seed}/{sub_fold}'\
                --output-dir-2 '/workspace/valid_results_ablation_study_100_s{current_seed}/{sub_fold}'")
        

## For testing
# Choose the best epoch from each fold
best_epoch = {'Fold_0': [41, 44, 21, 40, 26],
              'Fold_1': [20, 39, 38, 8, 16],
              'Fold_2': [48, 30, 25, 40, 49],
              'Fold_3': [25, 19, 37, 26, 12],
              'Fold_4': [36, 31, 44, 45, 39]}


for current_fold, current_seed in zip(fold_list, seed_list):

    current_best_epoch = best_epoch[current_fold]

    os.system(f"CUDA_VISIBLE_DEVICES='0' python test_k_fold.py --data-folder-dir '/workspace/{current_fold}'\
            --model-name 'Up_SMART_Net_Dual_CLS_REC' --num-workers 10 --pin-mem --training-stream 'Upstream' --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0'\
            --from-pretrained-0 '/workspace/checkpoints_ablation_study_100_s{current_seed}/Fold_0/epoch_{current_best_epoch[0]}_checkpoint.pth'\
            --from-pretrained-1 '/workspace/checkpoints_ablation_study_100_s{current_seed}/Fold_1/epoch_{current_best_epoch[1]}_checkpoint.pth'\
            --from-pretrained-2 '/workspace/checkpoints_ablation_study_100_s{current_seed}/Fold_2/epoch_{current_best_epoch[2]}_checkpoint.pth'\
            --from-pretrained-3 '/workspace/checkpoints_ablation_study_100_s{current_seed}/Fold_3/epoch_{current_best_epoch[3]}_checkpoint.pth'\
            --from-pretrained-4 '/workspace/checkpoints_ablation_study_100_s{current_seed}/Fold_4/epoch_{current_best_epoch[4]}_checkpoint.pth'\
            --checkpoint-total-num 5 --print-freq 1 --output-dir '/workspace/checkpoints_ablation_study_100_s{current_seed}/test'")



## for Guided GradCAM (example for Fold_4 (seed 915))
os.system("CUDA_VISIBLE_DEVICES='0' python inference.py --data-folder-dir '/workspace/Fold_4'\
          --model-name 'Up_SMART_Net_Dual_CLS_REC' --num-workers 20 --pin-mem --training-stream 'Upstream' --multi-gpu-mode 'DataParallel' --cuda-visible-devices '0'\
          --resume '/workspace/checkpoints_ablation_study_100_s915/Fold_1/epoch_89_checkpoint.pth'\
          --print-freq 1 --output-dir '/workspace/checkpoints_ablation_study_100_s915/activation_map/Fold_1'")




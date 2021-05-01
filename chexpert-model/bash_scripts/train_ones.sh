SAVE_DIR=$1
WANDB_PROJECT_NAME=$2

UNCERTAIN_MAP_PATH='/deep/group/CheXpert/Uncertainty/uncertainty_ones.csv'

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_ones_1 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name uncertainty_ones_1_train \
                    --uncertain_map_path $UNCERTAIN_MAP_PATH \

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_ones_2 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name uncertainty_ones_2_train \
                    --uncertain_map_path $UNCERTAIN_MAP_PATH \

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_ones_3 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name uncertainty_ones_3_train \
                    --uncertain_map_path $UNCERTAIN_MAP_PATH \
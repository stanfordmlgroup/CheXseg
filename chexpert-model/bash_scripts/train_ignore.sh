SAVE_DIR=$1
WANDB_PROJECT_NAME=$2

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_ignore_1 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name u_ignore_1_train \

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_ignore_2 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name u_ignore_2_train \

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_ignore_3 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name u_ignore_3_train \
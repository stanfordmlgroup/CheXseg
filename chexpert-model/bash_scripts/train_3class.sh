SAVE_DIR=$1
WANDB_PROJECT_NAME=$2

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_3class_1 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name u_3class_1_train \
                    --model_uncertainty True

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_3class_2 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name u_3class_2_train \
                    --model_uncertainty True

python3 ../train.py --dataset chexpert \
                    --batch_size 16 \
                    --experiment_name u_3class_3 \
                    --num_epochs 3 \
                    --scale 320 \
                    --save_dir $SAVE_DIR \
                    --save_top_k 10 \
                    --wandb_project_name $WANDB_PROJECT_NAME \
                    --wandb_run_name u_3class_3_train \
                    --model_uncertainty True
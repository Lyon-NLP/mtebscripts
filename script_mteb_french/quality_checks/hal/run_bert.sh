seeds=(
    42
    2024
    5050
)

for seed in "${seeds[@]}";
do
    python hal_baseline_bert.py --dataset_seed $seed --epochs 5 --batch_size 32 --lr 1e-4
done

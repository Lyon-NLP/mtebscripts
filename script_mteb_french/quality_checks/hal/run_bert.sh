seeds=(
    # 42
    2024
    5050
)

cd script_mteb_french/quality_checks/hal

for seed in "${seeds[@]}";
do
    python hal_baseline_bert.py --dataset_seed $seed
done

cd ../../..
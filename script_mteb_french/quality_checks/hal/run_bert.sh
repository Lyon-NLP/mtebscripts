seeds=(
    0
    42
    2024
)

cd script_mteb_french/quality_checks/hal

for seed in "${seeds[@]}";
do
    python hal_baseline_bert.py --model_seed $seed
done

cd ../../..
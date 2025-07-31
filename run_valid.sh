export WANDB_API_KEY="5409d3b960b01b25cec0f6abb5361b4022f0cc41"

python examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py \
	--project "wan2.2_finetune" \
	--data_csv "data/drones/meta.csv" \
	--dataset_name "drones" \
	--model_path "models/train/Wan2.2-TI2V-5B_lora/epoch-0-600.safetensors" \
	--max_count 10 \
	--use_lora False

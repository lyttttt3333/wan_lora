export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY="5409d3b960b01b25cec0f6abb5361b4022f0cc41"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/drones \
  --dataset_metadata_path data/drones/meta.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  # --resume_path "models/train/Wan2.2-TI2V-5B_lora/epoch-0-300"

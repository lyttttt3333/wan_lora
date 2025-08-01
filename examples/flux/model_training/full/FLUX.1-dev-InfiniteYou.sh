accelerate launch --config_file examples/flux/model_training/full/accelerate_config.yaml examples/flux/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_infiniteyou.csv \
  --data_file_keys "image,controlnet_image,infinityou_id_image" \
  --max_pixels 1048576 \
  --dataset_repeat 400 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,ByteDance/InfiniteYou:infu_flux_v1.0/aes_stage2/image_proj_model.bin,ByteDance/InfiniteYou:infu_flux_v1.0/aes_stage2/InfuseNetModel/*.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe." \
  --output_path "./models/train/FLUX.1-dev-InfiniteYou_full" \
  --trainable_models "controlnet,image_proj_model" \
  --extra_inputs "controlnet_image,infinityou_id_image,infinityou_guidance" \
  --use_gradient_checkpointing

import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import pandas

def main(args)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.load_lora(pipe.dit, args.model_path, alpha=1)
    pipe.enable_vram_management()
    
    df = pd.read_csv(args.data_csv) 
    
    wandb.init(project=args.project)
    
    count = 0
    for video_name, prompt in zip(df["video"], df["prompt"]):
        count += 1
        input_image = VideoData(f"data/{args.dataset_name}/{prompt}", height=480, width=832)[0]
        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=input_image,
            num_frames=49,
            seed=1, tiled=True,
        )
        if not args.upload_wandb:
            save_video(video, "output_videos/{video_name}", fps=15, quality=5)
        else:
            wandb.log({f"video_{video_name}": wandb.Video(output_path, fps=15, format="mp4")})
    
        if count >= args.max_count:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos and upload to wandb")
    parser.add_argument("--project", type=str, default="wan2.2_finetune", help="wandb project name")
    parser.add_argument("--data_csv", type=str, default="data/drones/meta.csv", help="path to CSV with video and prompt columns")
    parser.add_argument("--dataset_name", type=str, default="drones", help="path to CSV with video and prompt columns")
    parser.add_argument("--model_path", type=str, default="models/train/Wan2.2-TI2V-5B_lora/epoch-1-600.safetensors", help="path to CSV with video and prompt columns")
    parser.add_argument("--max_count", type=int, default=10, help="maximum number of videos to generate and upload")
    parser.add_argument("--upload_wandb", type=bool, default=True, help="maximum number of videos to generate and upload")
    args = parser.parse_args()
    main(args)

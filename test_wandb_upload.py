import wandb

wandb.login(key="5409d3b960b01b25cec0f6abb5361b4022f0cc41")  # 替换为你的 token

# 初始化一个 wandb run
wandb.init(project="wan2.2_finetune", name="upload_test_run")

# 上传视频
video_path = "data/drones/fcfab9b7-24bc-46f2-9409-1d8ae306514d.mp4"  # 替换为你本地的视频路径
wandb.log({"example_video": wandb.Video(video_path, fps=15, format="mp4")})

# 结束 run
wandb.finish()

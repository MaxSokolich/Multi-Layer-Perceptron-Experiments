import cv2
import numpy as np

# 读取图片
image_path = 'media/drawing.png'
image = cv2.imread(image_path)

# 获取图片的宽度和高度
height, width, _ = image.shape

# 定义视频的帧率和时长
fps = 30
duration = 60  # seconds
frame_count = fps * duration

# 定义视频编码和创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = 'media/output_video2.mp4'
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# 将图片写入视频的每一帧
for _ in range(frame_count):
    video.write(image)

# 释放 VideoWriter 对象
video.release()

print(f"视频已保存到 {video_path}")
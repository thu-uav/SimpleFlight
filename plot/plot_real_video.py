# import cv2
# import numpy as np

# # 文件路径
# input_video_path = '/home/jiayu/OmniDrones/plot/normal_raw.mp4'
# output_video_path = '/home/jiayu/OmniDrones/plot/process_video.mp4'

# # 定义要裁剪的帧数
# T1 = 100  # 删除开头的T1帧
# T2 = 800   # 删除结尾的T2帧

# # 打开视频文件
# cap = cv2.VideoCapture(input_video_path)
# if not cap.isOpened():
#     print("Can not open the file!")
#     exit()

# # 获取视频属性
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# # 创建空白的轨迹画布
# trajectory_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# # 定义亮度阈值，提取无人机位置
# brightness_threshold = 200

# # 跳过开头的T1帧
# for _ in range(T1):
#     ret, frame = cap.read()
#     if not ret:
#         break

# # 读取剩余的帧并处理
# frame_index = T1
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 如果帧数超过总帧数减去T2，则跳过
#     if frame_index >= frame_count - T2:
#         frame_index += 1
#         continue

#     # 转换为灰度图
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 检测亮光部分
#     _, binary_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

#     # 在轨迹画布中保留原始像素（更新轨迹画布，但保留已有的像素）
#     nonzero_indices = np.where(binary_mask > 0)
#     trajectory_canvas[nonzero_indices[0], nonzero_indices[1]] = frame[nonzero_indices[0], nonzero_indices[1]]

#     # 将轨迹画布完全叠加到当前帧
#     overlay_frame = frame.copy()
#     mask = (trajectory_canvas > 0).any(axis=-1)
#     overlay_frame[mask] = trajectory_canvas[mask]

#     # 写入输出视频
#     out.write(overlay_frame)

#     frame_index += 1

# # 释放资源
# cap.release()
# out.release()

# print("处理完成，轨迹视频已保存为:", output_video_path)


import cv2
import numpy as np

# 文件路径
input_video_path = '/home/jiayu/OmniDrones/plot/fast_raw.mp4'
output_video_path = '/home/jiayu/OmniDrones/plot/fast.mp4'

# 定义要裁剪的帧数
T1 = 180  # 删除开头的T1帧
T2 = 100   # 删除结尾的T2帧

# 定义向下平移的像素数
N = 50

# 打开视频文件
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Can not open the file!")
    exit()

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 创建空白的轨迹画布
trajectory_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# 定义亮度阈值，提取无人机位置
brightness_threshold = 150

# 跳过开头的T1帧
for _ in range(T1):
    ret, frame = cap.read()
    if not ret:
        break

# 读取剩余的帧并处理
frame_index = T1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 如果帧数超过总帧数减去T2，则跳过
    if frame_index >= frame_count - T2:
        frame_index += 1
        continue

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测亮光部分
    _, binary_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # 在轨迹画布中保留原始像素（更新轨迹画布，但保留已有的像素）
    nonzero_indices = np.where(binary_mask > 0)
    trajectory_canvas[nonzero_indices[0], nonzero_indices[1]] = frame[nonzero_indices[0], nonzero_indices[1]]

    # 将轨迹画布完全叠加到当前帧
    overlay_frame = frame.copy()
    mask = (trajectory_canvas > 0).any(axis=-1)
    overlay_frame[mask] = trajectory_canvas[mask]

    # 向下平移N个像素
    shifted_frame = np.zeros_like(overlay_frame)
    shifted_frame[N:, :] = overlay_frame[:-N, :]  # 向下平移
    shifted_frame[:N, :] = overlay_frame[-N:, :]  # 将下方的像素补在上方

    # 写入输出视频
    out.write(shifted_frame)

    frame_index += 1

# 释放资源
cap.release()
out.release()

print("处理完成，轨迹视频已保存为:", output_video_path)
"""
show_3d_depth.py - 直接显示 3D 点云
"""

import airsim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("正在连接 AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("连接成功！")

# 获取图像
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
])

# 解析 RGB
rgb = None
if responses[0].width > 0:
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

# 解析深度图
depth = np.array(responses[1].image_data_float, dtype=np.float32)
depth = depth.reshape(responses[1].height, responses[1].width)

print(f"深度图尺寸: {depth.shape}")
print(f"深度范围: {depth.min():.2f}m - {depth.max():.2f}m")

# 创建点云
h, w = depth.shape
subsample = 4
max_depth = 25.0

depth_sub = depth[::subsample, ::subsample]
h_sub, w_sub = depth_sub.shape

# 相机参数
fov_deg = 90
fov_rad = np.radians(fov_deg)
fx = w / (2 * np.tan(fov_rad / 2))
fy = fx
cx, cy = w / 2, h / 2

# 像素坐标
u = np.arange(0, w, subsample)
v = np.arange(0, h, subsample)
u, v = np.meshgrid(u, v)

z = depth_sub.flatten()
valid = (z > 0) & (z < max_depth)
z = z[valid]
u = u.flatten()[valid]
v = v.flatten()[valid]

# 3D 坐标
x = (u - cx) * z / fx
y = (v - cy) * z / fy

print(f"点云包含 {len(x)} 个点")

# 绘图
fig = plt.figure(figsize=(15, 5))

# RGB
ax1 = fig.add_subplot(131)
if rgb is not None:
    ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
ax1.set_title("RGB Image")
ax1.axis('off')

# 深度热力图
ax2 = fig.add_subplot(132)
im = ax2.imshow(depth, cmap='turbo', vmin=0, vmax=25)
ax2.set_title("Depth Map")
plt.colorbar(im, ax=ax2, label='Distance (m)')
ax2.axis('off')

# 3D 点云
ax3 = fig.add_subplot(133, projection='3d')
scatter = ax3.scatter(x, z, -y, c=z, cmap='turbo', s=1, alpha=0.6)
ax3.set_xlabel('X (left-right)')
ax3.set_ylabel('Z (depth)')
ax3.set_zlabel('Y (up-down)')
ax3.set_title("3D Point Cloud")
ax3.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.savefig("depth_3d_view.png", dpi=150)
print("\n已保存: depth_3d_view.png")
plt.show()

import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#flow_path = "2011_09_26_drive_0101_sync/flow/0000000116.png"
flow_path = "2011_09_26_drive_0101_sync/flow/0000000373.png"
flow = cv2.imread(flow_path, -1)
# flow = flow.astype(np.float64)
flow_u = flow[:, :, 2].astype(np.float32)
flow_v = flow[:, :, 1].astype(np.float32)
flow_u = (flow_u - 2 ** 15) / 64.0
flow_v = (flow_v - 2 ** 15) / 64.0

# plt.subplot(211)
# plt.imshow(flow_u, cmap="RdYlGn"), plt.axis('off')
# plt.subplot(212)
plt.imshow(flow_v, cmap="RdYlGn"), plt.axis('off')
plt.show()

vmax = flow_v.max()
vmin = flow_v.min()
VImage = np.zeros((flow_v.shape[0], vmax - vmin + 1))
VImage = VImage.astype(np.uint8)
for i in np.arange(flow_v.shape[0]):
    for j in flow_v[i]:
        VImage[i, j-vmin] += 1

plt.imshow(VImage), plt.axis('off')
plt.show()

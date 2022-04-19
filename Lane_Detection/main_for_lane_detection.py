import math
import os
import cv2
import numpy as np
from utilities import RANSAC, Edge_Detection
from PIL import Image
from matplotlib import pyplot as plt
from scipy.linalg import lstsq

# get paths

pic_num = 739  # 635
# pic_num = 116 330 331 410 411 532 560 561 699 700 875 876
path = "E:\\Program Files\\dataset\\KITTI\\2011_09_26_drive_0101_sync"
original_img_path = path + f"\\image_2\\0000000{pic_num}.png"
disparity_path = path + f"\\disparity\\0000000{pic_num}.png"
semantic_path = path + f"\\semantic\\0000000{pic_num}.png"

# read original left image and disparity
rgb_img = cv2.imread(original_img_path, 1)
disparity_img = cv2.imread(disparity_path, 0)

plt.figure("original & disparity")
plt.subplot(2, 1, 1)
plt.imshow(rgb_img)
plt.title("original")
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(disparity_img)
plt.title("disparity")
plt.axis('off')

# read mask
semantic_img = cv2.imread(semantic_path)
mask = cv2.inRange(semantic_img, (0, 0, 255), (0, 0, 255))
mask = mask / 255
mask = mask.astype(np.uint8)

disparity_img = disparity_img * mask
disparity_img.astype(np.uint8)
MaxDisp = round(disparity_img.max())

cols = np.size(disparity_img, 1)  # width
rows = np.size(disparity_img, 0)  # height

# -------------------------------------------------------------------
#                   Compute VDisparity
# -------------------------------------------------------------------

VDispImage = np.zeros((rows, MaxDisp + 1), dtype=np.uint8)

for i in np.arange(rows):
    for j in disparity_img[i, :]:
        if j < MaxDisp:
            VDispImage[i, round(j)] += 1

# -------------------------------------------------------------------
#                    VDisparity dynamic programming
# -------------------------------------------------------------------

# dynamic programming
dmax = VDispImage[-1].argmax()
# Energy = np.zeros(( VDispImage.shape[0], dmax + 1), dtype=int)
lamda = 20  # lambda parameter for wight of connectivity

state_v = VDispImage.shape[0] - 1
state_lst = [[state_v, dmax]]
pre_action = 0
for d in np.arange(dmax - 1, -1, -1):
    best_action = 0
    reward_max = 0
    for action in np.arange(0, 8):
        reward = int(VDispImage[state_v - action, d]) + \
                 abs(pre_action - action) * (-2) + action * (-2)
        if reward > reward_max:
            best_action = action
            reward_max = reward
    state_v -= best_action
    state_lst.append([state_v, d])

OptimalSln = np.array(state_lst)  # action point

# RANSAC to fit Parabola
beta = RANSAC(OptimalSln)  # parabola coefficient

# compute Vanishing_y
Vanishing_y_compute = lambda v, beta: int(v - (beta[2] + beta[1] * v + beta[0] * v ** 2) / (beta[1] + 2 * beta[0] * v))
Vanishing_y = np.array([Vanishing_y_compute(v, beta) for v in range(VDispImage.shape[0])])

v0 = 173
disp_est = np.polyval(beta, np.arange(v0, rows))

plt.figure("v disparity")
plt.imshow(VDispImage)
plt.plot(disp_est, np.arange(v0, rows), 'r')
plt.title("v disparity")
plt.axis('off')

# -------------------------------------------------------------------
#                    get edges
# -------------------------------------------------------------------

rgb_img = cv2.bilateralFilter(src=rgb_img, d=10, sigmaColor=0.3, sigmaSpace=300)  # bilateral filter

edges = cv2.Canny(rgb_img, 150, 200, apertureSize=3)  # edge detector

Vanishing_y0 = int((- beta[1] + math.sqrt(beta[1] ** 2 - 4 * beta[2] * beta[0])) / (
        2 * beta[0]))
Road_Height = rows - Vanishing_y0

edges = edges * mask
plt.figure("Road")
plt.subplot(2, 1, 1)
plt.imshow(rgb_img)
plt.title("road with filter")
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(edges)
plt.title("road edges")
plt.axis('off')

# -------------------------------------------------------------------
#                    sparse Vpx 
# -------------------------------------------------------------------


Vanishing_x_Accumulator = np.zeros((int(Road_Height), cols))
Orientation_Accumulator = np.zeros((rows, cols))

grad_x, grad_y = Edge_Detection(Sobel_Image=edges, threshold=0, height=rows, width=cols,
                                Vanishing_y0=Vanishing_y0,
                                Vanishing_x_Accumulator=Vanishing_x_Accumulator,
                                Vanishing_y=Vanishing_y,
                                Orientation_Accumulator=Orientation_Accumulator)
grad_x = grad_x * mask.astype(np.float)
grad_y = grad_y * mask.astype(np.float)
Vanishing_x_Accumulator = Vanishing_x_Accumulator * mask[Vanishing_y0:]

plt.figure("Sparse Vpx")
plt.imshow(Vanishing_x_Accumulator)
plt.axis('off')

plt.figure("gradients")
plt.subplot(2, 1, 1)
plt.imshow(grad_y)
plt.title("horizontal")
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(grad_x)
plt.title("vertical")
plt.axis('off')

# -------------------------------------------------------------------
#                    dense Vpx -- Noise_Elimination
# -------------------------------------------------------------------

Band_Size = 70
Band = np.zeros((Band_Size, cols))
Vp_X_Dynamic_Map_Final = np.zeros((Road_Height, cols))

for i in np.arange(Road_Height - 1, -1, -1):
    if i + 1 >= Road_Height:
        for j in np.arange(i, i - Band_Size, -1):
            for m in np.arange(cols):
                k = j - Road_Height + Band_Size
                Band[k, m] = Vanishing_x_Accumulator[j, m]

        for q in np.arange(cols):
            T = 0
            for p in np.arange(Band_Size):
                T += Band[p, q]
            Vp_X_Dynamic_Map_Final[i, q] = T

    if i + 1 < Road_Height and i - Band_Size >= 0:
        for j in np.arange(cols):
            Vp_X_Dynamic_Map_Final[i, j] = Vp_X_Dynamic_Map_Final[i + 1, j] - \
                                           Vanishing_x_Accumulator[i + 1, j] + \
                                           Vanishing_x_Accumulator[i - Band_Size + 1, j]

    if i - Band_Size < 0:
        for j in np.arange(cols):
            Vp_X_Dynamic_Map_Final[i, j] = Vp_X_Dynamic_Map_Final[i + 1, j] - \
                                           Vanishing_x_Accumulator[i + 1, j]

# -------------------------------------------------------------------
#                    dense Vpx -- Dynamic
# -------------------------------------------------------------------

Vp_X = np.zeros(cols)
Vp_X_Accumulator = np.zeros(cols)
Vp_Maximize_Array = np.zeros(cols)
Vp_X_Dynamic_Map = np.zeros((Road_Height, cols))
Vp_X_Position = np.zeros((Road_Height, 2))
Vp_Dynamic_Accumulator = np.zeros((9, cols))

for i in np.arange(cols):
    Vp_X[i] = Vp_X_Dynamic_Map_Final[Road_Height - 1, i]
    Vp_X_Accumulator[i] = Vp_X_Dynamic_Map_Final[Road_Height - 2, i]

for k in np.arange(Road_Height - 1, 2, -1):

    lambda_1 = 1.07  # pre-set parameter
    # lambda_1 = 0.87  # pre-set parameter

    for i in np.arange(-4, 5):
        for j in np.arange(cols):
            if cols > j + i >= 0:
                Vp_Dynamic_Accumulator[i + 4, j] = int(Vp_X_Accumulator[j] + lambda_1 * Vp_X[j + i])
            else:
                Vp_Dynamic_Accumulator[i + 4, j] = 0

    for i in np.arange(cols):

        Vp_X_Accumulator[i] = Vp_X_Dynamic_Map_Final[k - 3, i]

        Maximize1 = 0
        for j in np.arange(9):
            if Vp_Dynamic_Accumulator[j, i] > Maximize1:
                Vp_X[i] = Maximize1 = Vp_Dynamic_Accumulator[j, i]
                Vp_Maximize_Array[i] = j
                Vp_X_Dynamic_Map[k, i] = Vp_Maximize_Array[i]
    # Count+=1

    Maximize2 = 0
    V_P_X = 0
    for t in np.arange(cols):
        if Vp_X[t] >= Maximize2:
            Maximize2 = Vp_X[t]
            V_P_X = t

    Vp_X_Position[k, 0] = V_P_X
    Vp_X_Position[k, 1] = k

plt.figure("Dense Vpx")
plt.imshow(Vp_X_Dynamic_Map_Final)
plt.plot(Vp_X_Position[:, 0], Vp_X_Position[:, 1], 'm')
plt.axis('off')

# solve quartic polynomial
Y = np.zeros(Road_Height - 3)
XX = np.zeros((Road_Height - 3, 5))
# Coeff = np.zeros(5)
for i in np.arange(3, Road_Height):
    M = Vp_X_Position[i, 0]
    N = Vp_X_Position[i, 1]
    Y[i - 3] = M
    XX[i - 3, 0] = N ** 4
    XX[i - 3, 1] = N ** 3
    XX[i - 3, 2] = N ** 2
    XX[i - 3, 3] = N
    XX[i - 3, 4] = 1

Coeff, res, rnk, s = lstsq(XX, Y)
gamma4 = Coeff[0]
gamma3 = Coeff[1]
gamma2 = Coeff[2]
gamma1 = Coeff[3]
gamma0 = Coeff[4]

# Drawing the Vanishing_x Accumulator
n = np.arange(Road_Height - 1, 2, -1)
m = gamma4 * n * n * n * n + gamma3 * n * n * n + gamma2 * n * n + gamma1 * n + gamma0
plt.plot(m, n, 'r')

# -------------------------------------------------------------------
#                    DrawLane
# -------------------------------------------------------------------

# Detecting the three lanes and drawing them out

Curve_Accumulator = np.zeros((rows, cols))
Amplitude_Array = np.zeros(cols)

Shift_Size = 20
# Shift_Size = 12

for i in np.arange(Shift_Size, cols * 3 // 4):

    Total_Amplitude = 0
    j = Road_Height - 1

    Vanishing_x = sum(Coeff * np.array([j ** 4, j ** 3, j ** 2, j, 1]))

    Next_Column = round((Vanishing_x - i * (1 - j)) / j)
    First_Column = Next_Column

    for j in np.arange(Road_Height - 2, 0, -1):
        Max = 0
        Min = 100000

        shift_band = Orientation_Accumulator[j, First_Column - Shift_Size:First_Column - Shift_Size + 1]
        Min = min(Min, shift_band.min())
        Max = min(Max, shift_band.max())

        Amplitude_Value = Max - Min
        Total_Amplitude = Total_Amplitude + Amplitude_Value

        # Calculate the next row
        Vanishing_x = sum(Coeff * np.array([j ** 4, j ** 3, j ** 2, j, 1]))

        Next_Column = round((Vanishing_x - First_Column * (1 - j)) / j)
        First_Column = Next_Column
        Curve_Accumulator[(j + Vanishing_y0), i] = Next_Column

    Amplitude_Array[i] = Total_Amplitude

plt.figure("Amplitude_Array")
# plt.imshow(Amplitude_Array)
plt.plot(Amplitude_Array)

plt.axis('off')

# Find Three Maximize and Draw the Lane
Erase_Range = 10
Lane_Position = 0
for t in np.arange(3):

    Max = 0
    for m in np.arange(cols * 3 // 4):

        if Amplitude_Array[m] >= Max:
            Max = Amplitude_Array[m]
            Lane_Position = m

        # Erase the area
        Amplitude_Array[Lane_Position - Erase_Range:Lane_Position - Erase_Range + 1] = 0

    # for n in np.arange( rows - 2,  Vanishing_y0 + 30, -1):
    #     z1 = Curve_Accumulator[n, Lane_Position]
    #     z2 = Curve_Accumulator[(n - 1), Lane_Position]
    #
    #     plt.figure("draw lane")
    #     plt.imshow( rgb_img)
    #     plt.plot([z1, z2], [n, n], 'r')
    #
    #     plt.axis('off')
    color = ['r', 'm', 'b']
    plt.figure("draw lane")
    plt.imshow(rgb_img)
    plt.plot(Curve_Accumulator[Vanishing_y0 + 30: rows - 1, Lane_Position],
             np.arange(Vanishing_y0 + 30, rows - 1), color[t])

    plt.axis('off')

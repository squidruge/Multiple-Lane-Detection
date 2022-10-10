import math
import os
import cv2
import numpy as np
from scipy import signal

from utilities import RANSAC, Edge_Detection
from PIL import Image
from matplotlib import pyplot as plt
from scipy.linalg import lstsq

# get paths

pic_num = 420  # 640  # 703  # 861  # 703  # 861  # 400  # 682  # 861  # 739  # 922
# pic_num = 116 330 331 410 411 532 560 561 699 700 875 876
plt.close('all')
for xxx in np.arange(1):
    pic_num = pic_num + xxx
    # path = "E:\\Program Files\\dataset\\KITTI\\2011_10_03_drive_0047_sync"
    path = "E:\\Program Files\\dataset\\KITTI\\2011_09_26_drive_0101_sync"
    original_img_path = path + f"\\image_2\\0000000{pic_num}.png"
    disparity_path = path + f"\\disparity\\0000000{pic_num}.png"
    semantic_path = path + f"\\semantic\\0000000{pic_num}.png"
    optical_flow_path = path + f"\\flow\\0000000{pic_num}.png"
    fig_save_path = "E:\\Program Files\\Lane_Detection_Vp\\Lane_Detection\\figs\\"

    # -------------------------------------------------------------------
    #                  optical_flow
    # -------------------------------------------------------------------
    optical_flow_img = cv2.imread(optical_flow_path, -1)
    # valid = optical_flow_img[:, :, 0]
    fu = optical_flow_img[:, :, 2].astype(np.float32)
    fv = optical_flow_img[:, :, 1].astype(np.float32)
    fu = (fu - 2 ** 15) / 64.0
    fv = (fv - 2 ** 15) / 64.0
    # fu[valid == 0] = np.nan
    # fv[valid == 0] = np.nan

    # read original left image and disparity
    rgb_img = cv2.imread(original_img_path, 1)
    rgb_img_origin=rgb_img
    disparity_img = cv2.imread(disparity_path, 0)

    plt.figure("original & disparity %d" % pic_num)
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
    #                   PT
    # -------------------------------------------------------------------
    # d = destinationx
    # vmax = img1.shape[0]  # v
    # umax = img1.shape[1]  # u
    # v_map_1 = np.mat(np.arange(0, vmax))  #
    # v_map_1_transpose = v_map_1.T  # (1030, 1)
    # umax_one = np.mat(np.ones(umax)).astype(int)  # (1, 1720) 元素设置为1
    # v_map = v_map_1_transpose * umax_one  # (1030, 1720)
    # vmax_one = np.mat(np.ones(vmax)).astype(int)
    # vmax_one_transpose = vmax_one.T  # (1030, 1)
    # u_map_1 = np.mat(np.arange(0, umax))  # (1, 1720)
    # u_map = vmax_one_transpose * u_map_1  # (1030, 1720)
    # Su = np.sum(u)
    # Sv = np.sum(v)
    # Sd = np.sum(d)
    # Su2 = np.sum(np.square(u))
    # Sv2 = np.sum(np.square(v))
    # Sdu = np.sum(np.multiply(u, d))
    # Sdv = np.sum(np.multiply(v, d))
    # Suv = np.sum(np.multiply(u, v))
    # n = len(u)
    # beta0 = (np.square(Sd) * (Sv2 + Su2) - 2 * Sd * (Sv * Sdv + Su * Sdu) + n * (np.square(Sdv) + np.square(Sdu))) / 2
    # beta1 = (np.square(Sd) * (Sv2 - Su2) + 2 * Sd * (Su * Sdu - Sv * Sdv) + n * (np.square(Sdv) - np.square(Sdu))) / 2
    # beta2 = -np.square(Sd) * Suv + Sd * (Sv * Sdu + Su * Sdv) - n * Sdv * Sdu
    # gamma0 = (n * Sv2 + n * Su2 - np.square(Sv) - np.square(Su)) / 2
    # gamma1 = (n * Sv2 - n * Su2 - np.square(Sv) + np.square(Su)) / 2
    # gamma2 = Sv * Su - n * Suv
    # A = (beta1 * gamma0 - beta0 * gamma1)
    # B = (beta0 * gamma2 - beta2 * gamma0)
    # C = (beta1 * gamma2 - beta2 * gamma1)
    # delta = np.square(A) + np.square(B) - np.square(C)
    # tmp1 = (A + np.sqrt(delta)) / (B - C)
    # tmp2 = (A - np.sqrt(delta)) / (B - C)
    # theta1 = math.atan(tmp1)
    # theta2 = math.atan(tmp2)
    # u = np.mat(u)
    # v = np.mat(v)
    # d = np.mat(d)
    # d = d.T
    # u = u.T
    # v = v.T
    # t1 = v * math.cos(theta1) - u * math.sin(theta1)
    # t2 = v * math.cos(theta2) - u * math.sin(theta2)
    # n_ones = np.ones(n).astype(int)
    # n_ones = (np.mat(n_ones)).T
    # T1 = np.hstack((n_ones, t1))
    # T2 = np.hstack((n_ones, t2))
    # f1 = d.T * T1 * np.linalg.inv(T1.T * T1) * T1.T * d
    # f2 = d.T * T2 * np.linalg.inv(T2.T * T2) * T2.T * d
    # if f1 < f2:
    #     theta = theta2
    # else:
    #     theta = theta1
    # t = v * math.cos(theta) - u * math.sin(theta)
    # T = np.hstack((n_ones, t))
    # a = np.linalg.inv(T.T * T) * T.T * d
    # # print("a[0]:%f    a[1]:%f    theta:%f"%(a[0],a[1],theta))
    # randomtheta = random.uniform(-0.1, 0.1)  # 加入扰动
    # theta1 = theta + randomtheta  #
    # # remap
    # new_right = np.zeros_like(img2c)
    # t_map = v_map * math.cos(theta1) - u_map * math.sin(theta1)
    # fd = (a[0] + np.multiply(a[1], t_map)) - 15
    # u, v = np.meshgrid(u_map_1, v_map_1)
    # u = np.float32(u - fd)
    # v = np.float32(v)
    # new_right = cv2.remap(img2c, u, v, cv2.INTER_LINEAR)
    # newdisp = fd.copy()
    # disp = depth2disp(depth)
    # newdisp = disp - fd

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
    Vanishing_y_compute = lambda v, beta: int(
        v - (beta[2] + beta[1] * v + beta[0] * v ** 2) / (beta[1] + 2 * beta[0] * v))
    Vanishing_y = np.array([Vanishing_y_compute(v, beta) for v in range(VDispImage.shape[0])])

    v0 = 173
    disp_est = np.polyval(beta, np.arange(v0, rows))

    plt.figure("v disparity %d" % pic_num)
    plt.imshow(VDispImage)
    plt.plot(OptimalSln[:, 1], OptimalSln[:, 0], 'b')
    plt.plot(disp_est, np.arange(v0, rows), 'r')
    plt.title("v disparity %d" % pic_num)
    plt.axis('off')

    # -------------------------------------------------------------------
    #                    get edges
    # -------------------------------------------------------------------
    rgb_img = cv2.medianBlur(rgb_img, 5)
    rgb_img = cv2.bilateralFilter(src=rgb_img, d=12, sigmaColor=0.3, sigmaSpace=60)  # bilateral filter

    edges = cv2.Canny(rgb_img, 50, 200, apertureSize=3)  # edge detector

    Vanishing_y0 = int((- beta[1] + math.sqrt(beta[1] ** 2 - 4 * beta[2] * beta[0])) / (
            2 * beta[0])) + 20
    Road_Height = rows - Vanishing_y0

    edges = edges * mask
    plt.figure("Road %d" % pic_num)
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
    grad_x = grad_x * mask.astype(np.float64)
    grad_y = grad_y * mask.astype(np.float64)
    Vanishing_x_Accumulator = Vanishing_x_Accumulator * mask[Vanishing_y0:]

    plt.figure("Sparse Vpx %d" % pic_num)
    plt.imshow(Vanishing_x_Accumulator)
    plt.axis('off')
    #
    plt.figure("gradients %d" % pic_num)
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
    cut_band = 3

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

        # lambda_1 = 1.07  # pre-set parameter
        lambda_1 = 1.07  # pre-set parameter

        for i in np.arange(-4, 5):
            for j in np.arange(cols):
                if cols > j + i >= 0:
                    Vp_Dynamic_Accumulator[i + 4, j] = int(Vp_X_Accumulator[j] + lambda_1 * Vp_X[j + i])
                    # Vp_Dynamic_Accumulator[i + 4, j] = int(
                    #     Vp_X_Accumulator[j] + lambda_1 * Vp_X[j + i] / (0.0002 * abs(i) + 1))
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

    plt.figure("Dense Vpx %d" % pic_num)
    plt.imshow(Vp_X_Dynamic_Map_Final)
    plt.plot(Vp_X_Position[:, 0], Vp_X_Position[:, 1], 'm')
    plt.axis('off')

    # solve quartic polynomial

    Y = np.zeros(Road_Height - cut_band)
    XX = np.zeros((Road_Height - cut_band, 5))
    # Coeff = np.zeros(5)
    for i in np.arange(cut_band, Road_Height):
        M = Vp_X_Position[i, 0]
        N = Vp_X_Position[i, 1]
        Y[i - cut_band] = M
        XX[i - cut_band, 0] = N ** 4
        XX[i - cut_band, 1] = N ** 3
        XX[i - cut_band, 2] = N ** 2
        XX[i - cut_band, 3] = N
        XX[i - cut_band, 4] = 1

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
    # t = 0.5
    # k1, k2 = 1, 3
    #
    # M0 = np.zeros_like(disparity_img)
    # M1 = np.zeros_like(disparity_img)
    # H = np.zeros((int(2 * t + 1), cols))
    #
    # weight = np.zeros_like(grad_x)
    #
    # theta_e = np.arctan(grad_y / grad_x)
    # theta_p = np.zeros_like(weight)
    # sigma_g = 3.5
    # sigma_s = 3.14 / 36
    #
    # for i in np.arange(edges.shape[0]):
    #     temp = i - Vanishing_y0
    #     Vanishing_x = sum(Coeff * np.array([temp ** 4, temp ** 3, temp ** 2, temp, 1]))
    #     for j in np.arange(edges.shape[1]):
    #         if edges[i, j] != 0:
    #             theta_p[i, j] = np.arctan((i - Vanishing_y[i]) / (j - Vanishing_x))
    #             if abs(theta_e[i, j] - theta_p[i, j]) < 3.14 / 6:
    #                 weight[i, j] = np.exp(-abs(theta_e[i, j] - theta_p[i, j]) / ((sigma_g ** 2) * sigma_s))
    #
    # plt.figure("weight")
    # plt.imshow(weight)
    # plt.axis('off')
    #
    # for i in np.arange(k2, edges.shape[0] - k2):
    #     for j in np.arange(k1, edges.shape[1] - k1):
    #         M0[i, j] = np.sum(grad_x[i-k2:i+k2+1,j-k1:j+k1+1] * weight[i-k2:i+k2+1,j-k1:j+k1+1])
    #
    # plt.figure("M0")
    # plt.imshow(M0)
    # plt.axis('off')
    #
    # GX = np.array([
    #     [4, 3, 2, 1, 0, -1, -2, -3, -4],
    #     [5, 4, 3, 2, 0, -2, -3, -4, -5],
    #     [6, 5, 4, 3, 0, -3, -4, -5, -6],
    #     [7, 6, 5, 4, 0, -4, -5, -6, -7],
    #     [8, 7, 6, 5, 0, -5, -6, -7, -8],
    #     [7, 6, 5, 4, 0, -4, -5, -6, -7],
    #     [6, 5, 4, 3, 0, -3, -4, -5, -6],
    #     [5, 4, 3, 2, 0, -2, -3, -4, -5],
    #     [4, 3, 2, 1, 0, -1, -2, -3, -4]
    # ])
    #
    # M1 = signal.convolve2d(M0, GX, mode="same")
    # plt.figure("M1")
    # plt.imshow(M1)
    # plt.axis('off')

    # # Detecting the three lanes and drawing them out

    Curve_Accumulator = np.zeros((rows, cols))
    Amplitude_Array = np.zeros(cols)

    # Shift_Size = 20
    Shift_Size = 12

    for i in np.arange(Shift_Size, cols * 3 // 4):
        # for i in np.arange(Shift_Size, cols):

        Total_Amplitude = 0
        j = Road_Height - 1

        Vanishing_x = sum(Coeff * np.array([j ** 4, j ** 3, j ** 2, j, 1]))

        Next_Column = round((Vanishing_x - i * (1 - j)) / j)
        First_Column = Next_Column

        for j in np.arange(Road_Height - 2, 0, -1):
            # Max = 0
            # Min = 100000

            shift_band = Orientation_Accumulator[j + Vanishing_y0,
                         First_Column - Shift_Size:First_Column + Shift_Size + 1]
            # Min = min(Min, shift_band.min())
            # Max = min(Max, shift_band.max())
            Max = shift_band.max()
            Min = shift_band.min()

            Amplitude_Value = Max - Min
            Total_Amplitude = Total_Amplitude + Amplitude_Value

            # Calculate the next row
            Vanishing_x = sum(Coeff * np.array([j ** 4, j ** 3, j ** 2, j, 1]))

            Next_Column = round((Vanishing_x - First_Column * (1 - j)) / j)
            First_Column = Next_Column
            Curve_Accumulator[(j + Vanishing_y0), i] = Next_Column

        Amplitude_Array[i] = Total_Amplitude

    # plt.figure("Amplitude_Array")
    # # plt.imshow(Amplitude_Array)
    # plt.plot(Amplitude_Array)

    plt.axis('off')

    # Find Three Maximize and Draw the Lane
    Erase_Range = 180
    Lane_Position = 0

    plt.figure("draw lane %d" % pic_num)
    # estimate with flow
    if xxx != 0:
        lane_fu = np.zeros((3, rows - 1 - (Vanishing_y0_last + 30)))
        lane_fv = np.zeros((3, rows - 1 - (Vanishing_y0_last + 30)))
        for i in np.arange(Lane_Position_Last.shape[0]):
            for j in np.arange(Lane_Position_Last.shape[1]):
                lane_fu[i, j] = fu[Vanishing_y0_last + 30 + j, round(Lane_Position_Last[i, j])]
                lane_fv[i, j] = fv[Vanishing_y0_last + 30 + j, round(Lane_Position_Last[i, j])]

            plt.plot(Lane_Position_Last[i] + lane_fu[i],
                     np.arange(Vanishing_y0_last + 30, rows - 1) + lane_fv[i], 'b', linewidth=2.0)

    Lane_Position_Last = np.zeros((3, rows - 1 - (Vanishing_y0 + 30)))
    for t in np.arange(3):

        Max = 0
        for m in np.arange(cols):
            # for m in np.arange(cols * 3 // 4):

            if Amplitude_Array[m] >= Max:
                Max = Amplitude_Array[m]
                Lane_Position = m

        # Erase the area
        Amplitude_Array[Lane_Position - Erase_Range:Lane_Position + Erase_Range + 1] = 0

        # for n in np.arange( rows - 2,  Vanishing_y0 + 30, -1):
        #     z1 = Curve_Accumulator[n, Lane_Position]
        #     z2 = Curve_Accumulator[(n - 1), Lane_Position]
        #
        #     plt.figure("draw lane")
        #     plt.imshow( rgb_img)
        #     plt.plot([z1, z2], [n, n], 'r')
        #
        #     plt.axis('off')

        # color = ['r', 'm', 'b']

        plt.imshow(rgb_img_origin)

        plt.plot(Curve_Accumulator[Vanishing_y0 + 30: rows - 1, Lane_Position],
                 np.arange(Vanishing_y0 + 30, rows - 1), 'r', linewidth=1.5)
        plt.axis('off')

        # update

        Lane_Position_Last[t] = Curve_Accumulator[Vanishing_y0 + 30: rows - 1, Lane_Position]
    Vanishing_y0_last = Vanishing_y0
    plt.tight_layout()
    plt.savefig(fig_save_path + f"fig{pic_num}.png", dpi=600)
    # plt.figure("Orientation_Accumulator")
    # plt.imshow(Orientation_Accumulator)
    # plt.axis('off')

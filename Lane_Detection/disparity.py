import math
import os
import cv2
import numpy as np
from utilities import RANSAC, Edge_Detection
from PIL import Image
from matplotlib import pyplot as plt
from scipy.linalg import lstsq


class LaneDetect(object):
    # MaxDisp = 128  # 128?

    def __init__(self, original_img_path, disparity_path, semantic_path):

        self.rgb_img = cv2.imread(original_img_path, 1)
        self.disparity_img = cv2.imread(disparity_path, 0)

        plt.figure("original & disparity")
        plt.subplot(2, 1, 1)
        plt.imshow(self.rgb_img)
        plt.title("original")
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(self.disparity_img)
        plt.title("disparity")
        plt.axis('off')

        semantic_img = cv2.imread(semantic_path)
        self.mask = cv2.inRange(semantic_img, (0, 0, 255), (0, 0, 255))
        self.mask = self.mask / 255
        # self.mask = np.stack((self.mask, self.mask, self.mask), axis=2)
        self.mask = self.mask.astype(np.uint8)
        self.disparity_img = self.disparity_img * self.mask
        self.disparity_img.astype(np.uint8)

        self.MaxDisp = round(self.disparity_img.max())
        self.cols = np.size(self.disparity_img, 1)
        self.rows = np.size(self.disparity_img, 0)
        self.VDispImage = np.zeros((self.rows, self.MaxDisp + 1), dtype=np.uint8)
        self.UDispImage = np.zeros((self.MaxDisp + 1, self.cols), dtype=np.uint8)

    def ComputeUDisparity(self):

        for i in np.arange(self.cols):
            for j in self.disparity_img[:, i]:
                if j < self.MaxDisp:
                    self.UDispImage[j, i] += 1
        # cv2.imshow("U disparity", self.UDispImage)
        # cv2.waitKey(0)

    def ComputeVDisparity(self):

        for i in np.arange(self.rows):
            for j in self.disparity_img[i, :]:
                if j < self.MaxDisp:
                    self.VDispImage[i, round(j)] += 1
        # cv2.imshow("V disparity", self.VDispImage)
        # cv2.waitKey(0)

    def VEstimation(self):
        # dynamic programming
        dmax = self.VDispImage[-1].argmax()
        # Energy = np.zeros((self.VDispImage.shape[0], dmax + 1), dtype=int)
        lamda = 20  # lambda

        state_v = self.VDispImage.shape[0] - 1
        state_lst = [[state_v, dmax]]
        pre_action = 0
        for d in np.arange(dmax - 1, -1, -1):
            best_action = 0
            reward_max = 0
            for action in np.arange(0, 8):
                reward = int(self.VDispImage[state_v - action, d]) + \
                         abs(pre_action - action) * (-2) + action * (-2)
                if reward > reward_max:
                    best_action = action
                    reward_max = reward
            state_v -= best_action
            state_lst.append([state_v, d])

        OptimalSln = np.array(state_lst)

        # RANSAC to fit Parabola
        self.beta = RANSAC(OptimalSln)

        # Dense V estimation
        Vcompute = lambda v, beta: int(v - (beta[2] + beta[1] * v + beta[0] * v ** 2) / (beta[1] + 2 * beta[0] * v))
        self.Vanishing_y = np.array([Vcompute(v, self.beta) for v in range(self.VDispImage.shape[0])])

        v0 = 173
        disp_est = np.polyval(self.beta, np.arange(v0, self.rows))

        plt.figure("v disparity")
        plt.imshow(self.VDispImage)
        plt.plot(disp_est, np.arange(v0, self.rows), 'r')
        plt.title("v disparity")
        plt.axis('off')

    def RoadSurfaceEstimation(self):
        f_v = [np.polyval(self.beta, i) for i in np.arange(self.disparity_img.shape[0])]

        trRSE = 3  # pre-set in essay
        # img2 = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2RGB)  # too much time

        self.rgb_img = cv2.bilateralFilter(src=self.rgb_img, d=10, sigmaColor=0.3, sigmaSpace=300)  # bilateral filter

        # self.img_left=self.img_left * np.stack((self.mask, self.mask, self.mask), axis=2)
        # self.img_left = self.img_left * cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        # rgb_img_mask = np.array(self.rgb_img * self.mask)?
        # rgb_img_mask=rgb_img_mask.astype(np.uint8)
        self.edges = cv2.Canny(self.rgb_img, 150, 200, apertureSize=3)  # edge detector

        # img3 = np.zeros_like(self.rgb_img)
        self.Vanishing_y0 = int((-self.beta[1] + math.sqrt(self.beta[1] ** 2 - 4 * self.beta[2] * self.beta[0])) / (
                2 * self.beta[0]))
        self.Road_Height = self.rows - self.Vanishing_y0

        # for v in np.arange(self.disparity_img.shape[0]):
        #     for u in np.arange(self.disparity_img.shape[1]):
        #         if np.abs(self.disparity_img[v][u] - int(f_v[v])) <= trRSE and v >= self.Vanishing_y0:
        #             img2[int(v), int(u), 1] = min(255, img2[int(v), int(u), 1] + 60)  # get road surface
        #             img3[int(v), int(u)] = self.rgb_img[int(v), int(u)]  # reduced road surface in left image

        self.edges = self.edges * self.mask
        plt.figure("Road")
        plt.subplot(2, 1, 1)
        plt.imshow(self.rgb_img)
        plt.title("Road Surface")
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(self.edges)
        plt.title("Road edges")
        plt.axis('off')

    def UEstimation(self):
        # Sparse Vanshing x

        # g_u and g_v represent the vertical and horizontal gradients respectively
        # g_v = cv2.Sobel(self.edges, cv2.CV_64F, 1, 0)  # x
        # g_u = cv2.Sobel(self.edges, cv2.CV_64F, 0, 1)
        gray_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2GRAY)

        # g_v = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)  # x
        # g_u = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)

        self.Vanishing_x_Accumulator = np.zeros((int(self.Road_Height), self.cols))
        self.Orientation_Accumulator = np.zeros((self.rows, self.cols))

        grad_x, grad_y = Edge_Detection(Sobel_Image=self.edges, threshold=0, height=self.rows, width=self.cols,
                                        Vanishing_y0=self.Vanishing_y0,
                                        Vanishing_x_Accumulator=self.Vanishing_x_Accumulator,
                                        Vanishing_y=self.Vanishing_y,
                                        Orientation_Accumulator=self.Orientation_Accumulator)
        grad_x = grad_x * self.mask.astype(np.float)
        grad_y = grad_y * self.mask.astype(np.float)
        self.Vanishing_x_Accumulator = self.Vanishing_x_Accumulator * self.mask[self.Vanishing_y0:]

        plt.figure("Sparse Vpx")
        plt.imshow(self.Vanishing_x_Accumulator)
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

        # Dense Vanshing x
        self.Noise_Elimination()  # dense Vpx

        Vp_X = np.zeros(self.cols)
        Vp_X_Accumulator = np.zeros(self.cols)
        Vp_Maximize_Array = np.zeros(self.cols)
        Vp_X_Dynamic_Map = np.zeros((self.Road_Height, self.cols))
        Vp_X_Position = np.zeros((self.Road_Height, 2))
        Vp_Dynamic_Accumulator = np.zeros((9, self.cols))

        for i in np.arange(self.cols):
            Vp_X[i] = self.Vp_X_Dynamic_Map_Final[self.Road_Height - 1, i]
            Vp_X_Accumulator[i] = self.Vp_X_Dynamic_Map_Final[self.Road_Height - 2, i]

        for k in np.arange(self.Road_Height - 1, 2, -1):

            lambda_1 = 1.07  # pre-set parameter
            # lambda_1 = 0.87  # pre-set parameter

            for i in np.arange(-4, 5):
                for j in np.arange(self.cols):
                    if self.cols > j + i >= 0:
                        Vp_Dynamic_Accumulator[i + 4, j] = int(Vp_X_Accumulator[j] + lambda_1 * Vp_X[j + i])
                    else:
                        Vp_Dynamic_Accumulator[i + 4, j] = 0

            for i in np.arange(self.cols):

                Vp_X_Accumulator[i] = self.Vp_X_Dynamic_Map_Final[k - 3, i]

                Maximize1 = 0
                for j in np.arange(9):
                    if Vp_Dynamic_Accumulator[j, i] > Maximize1:
                        Vp_X[i] = Maximize1 = Vp_Dynamic_Accumulator[j, i]
                        Vp_Maximize_Array[i] = j
                        Vp_X_Dynamic_Map[k, i] = Vp_Maximize_Array[i]
            # Count+=1

            Maximize2 = 0
            V_P_X = 0
            for t in np.arange(self.cols):
                if Vp_X[t] >= Maximize2:
                    Maximize2 = Vp_X[t]
                    V_P_X = t

            Vp_X_Position[k, 0] = V_P_X
            Vp_X_Position[k, 1] = k

        plt.figure("Dense Vpx")
        plt.imshow(self.Vp_X_Dynamic_Map_Final)
        plt.plot(Vp_X_Position[:, 0], Vp_X_Position[:, 1], 'm')
        plt.axis('off')

        # solve quartic polynomial
        Y = np.zeros(self.Road_Height - 3)
        XX = np.zeros((self.Road_Height - 3, 5))
        # Coeff = np.zeros(5)
        for i in np.arange(3, self.Road_Height):
            M = Vp_X_Position[i, 0]
            N = Vp_X_Position[i, 1]
            Y[i - 3] = M
            XX[i - 3, 0] = N ** 4
            XX[i - 3, 1] = N ** 3
            XX[i - 3, 2] = N ** 2
            XX[i - 3, 3] = N
            XX[i - 3, 4] = 1

        self.Coeff, res, rnk, s = lstsq(XX, Y)
        gamma4 = self.Coeff[0]
        gamma3 = self.Coeff[1]
        gamma2 = self.Coeff[2]
        gamma1 = self.Coeff[3]
        gamma0 = self.Coeff[4]

        # Drawing the Vanishing_y Accumulator
        n = np.arange(self.Road_Height - 1, 2, -1)
        m = gamma4 * n * n * n * n + gamma3 * n * n * n + gamma2 * n * n + gamma1 * n + gamma0
        plt.plot(m, n, 'r')
        # for n in np.arange(self.Road_Height - 1, 2, -1):
        #     m1 = gamma4 * n * n * n * n + gamma3 * n * n * n + gamma2 * n * n + gamma1 * n + gamma0
        #     m2 = gamma4 * (n - 1) ** 4 + gamma3 * (n - 1) ** 3 + gamma2 * (n - 1) ** 2 + gamma1 * (n - 1) + gamma0
        #     plt.plot([m1, n], [m2, n], 'r')

    def Noise_Elimination(self):
        Band_Size = 70
        Band = np.zeros((Band_Size, self.cols))
        self.Vp_X_Dynamic_Map_Final = np.zeros((self.Road_Height, self.cols))
        for i in np.arange(self.Road_Height - 1, -1, -1):
            if i + 1 >= self.Road_Height:
                for j in np.arange(i, i - Band_Size, -1):
                    for m in np.arange(self.cols):
                        k = j - self.Road_Height + Band_Size
                        Band[k, m] = self.Vanishing_x_Accumulator[j, m]

                for q in np.arange(self.cols):
                    T = 0
                    for p in np.arange(Band_Size):
                        T += Band[p, q]
                    self.Vp_X_Dynamic_Map_Final[i, q] = T

            if i + 1 < self.Road_Height and i - Band_Size >= 0:
                for j in np.arange(self.cols):
                    self.Vp_X_Dynamic_Map_Final[i, j] = self.Vp_X_Dynamic_Map_Final[i + 1, j] - \
                                                        self.Vanishing_x_Accumulator[i + 1, j] + \
                                                        self.Vanishing_x_Accumulator[i - Band_Size + 1, j]

            if i - Band_Size < 0:
                for j in np.arange(self.cols):
                    self.Vp_X_Dynamic_Map_Final[i, j] = self.Vp_X_Dynamic_Map_Final[i + 1, j] - \
                                                        self.Vanishing_x_Accumulator[i + 1, j]

    def DrawLane(self):
        # Detecting the three lanes and drawing them out

        Curve_Accumulator = np.zeros((self.rows, self.cols))
        Amplitude_Array = np.zeros(self.cols)

        Shift_Size = 20
        # Shift_Size = 12

        for i in np.arange(Shift_Size, self.cols * 3 // 4):

            Total_Amplitude = 0
            j = self.Road_Height - 1

            Vanishing_x = sum(self.Coeff * np.array([j ** 4, j ** 3, j ** 2, j, 1]))

            Next_Column = round((Vanishing_x - i * (1 - j)) / j)
            First_Column = Next_Column

            for j in np.arange(self.Road_Height - 2, 0, -1):
                Max = 0
                Min = 100000

                shift_band = self.Orientation_Accumulator[j, First_Column - Shift_Size:First_Column - Shift_Size + 1]
                Min = min(Min, shift_band.min())
                Max = min(Max, shift_band.max())

                Amplitude_Value = Max - Min
                Total_Amplitude = Total_Amplitude + Amplitude_Value

                # Calculate the next row
                Vanishing_x = sum(self.Coeff * np.array([j ** 4, j ** 3, j ** 2, j, 1]))

                Next_Column = round((Vanishing_x - First_Column * (1 - j)) / j)
                First_Column = Next_Column
                Curve_Accumulator[(j + self.Vanishing_y0), i] = Next_Column

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
            for m in np.arange(self.cols * 3 // 4):

                if Amplitude_Array[m] >= Max:
                    Max = Amplitude_Array[m]
                    Lane_Position = m

                # Erase the area
                Amplitude_Array[Lane_Position - Erase_Range:Lane_Position - Erase_Range + 1] = 0

            # for n in np.arange(self.rows - 2, self.Vanishing_y0 + 30, -1):
            #     z1 = Curve_Accumulator[n, Lane_Position]
            #     z2 = Curve_Accumulator[(n - 1), Lane_Position]
            #
            #     plt.figure("draw lane")
            #     plt.imshow(self.rgb_img)
            #     plt.plot([z1, z2], [n, n], 'r')
            #
            #     plt.axis('off')
            color=['r','m','b']
            plt.figure("draw lane")
            plt.imshow(self.rgb_img)
            plt.plot(Curve_Accumulator[self.Vanishing_y0 + 30:self.rows - 1, Lane_Position],
                     np.arange(self.Vanishing_y0 + 30, self.rows - 1), color[t])

            plt.axis('off')

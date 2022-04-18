from disparity import LaneDetect

# get paths

pic_num = 739  # 635
# pic_num = 116 330 331 410 411 532 560 561 699 700 875 876
path = "E:\\Program Files\\dataset\\KITTI\\2011_09_26_drive_0101_sync"
original_img_path = path + f"\\image_2\\0000000{pic_num}.png"
disparity_path = path + f"\\disparity\\0000000{pic_num}.png"
semantic_path = path + f"\\semantic\\0000000{pic_num}.png"

LD = LaneDetect(disparity_path=disparity_path, original_img_path=original_img_path, semantic_path=semantic_path)

LD.ComputeVDisparity()

LD.VEstimation()
LD.RoadSurfaceEstimation()
LD.UEstimation()
LD.DrawLane()

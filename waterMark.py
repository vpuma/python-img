import os.path
import cv2
import math
import numpy as np

def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new

def add_water_mark(picfile, wmfile, location, wmtype=0):
    ''' add_water_mark: add watermark to a picture
    :param picfile: the picture file
    :param wmfile: the watermark image file( should be a png file)
    :param location: the location on the picture where to add the watermark (0-8)
                    # 0 1 2
                    # 3 4 5
                    # 6 7 8
    :param wmtype: 0(default)-根据设置的WM_TRANSPARENCY进行叠加  1-根据水印图片alpha通道的值进行叠加
    :return: 0-succeeded, 1-failed
    :output: a new picture file(adding "wm" to the end of input picture filename)
    '''

    WM_MARGIN_x = 10
    WM_MARGIN_Y = 10
    WM_TRANSPARENCY = 1 # 0.1-1.0 valid for type 1
    WM_AREA_PERCENTAGE = 2
    DEBUG = False

    # params validation
    if not picfile.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        print("# Bad picture")
        exit(-1)
    if not wmfile.lower().endswith('.png'):
        print("# Bad watermark picture")
        exit(-1)
    if not location in range(9):
        print("# Bad watermark location")
        exit(-1)
    if not wmtype in range(2):
        print("# Bad watermark mthod")
        exit(-1)

    if DEBUG:
        print("#" * 50)
        print("# Adding watermark to picture: {}".format(picfile))

    wm_type = wmtype
    img = cv2.imread(picfile)
    if wm_type == 1:
        watermark = cv2.imread(wmfile, cv2.IMREAD_UNCHANGED)
    else:
        watermark = cv2.imread(wmfile)
    wm_loc = location

    if wm_type == 1:
        if img.shape[2] == 3:
            img = add_alpha_channel(img)

    # resize watermark
    wm_scale = math.sqrt((WM_AREA_PERCENTAGE / 100) / ((watermark.shape[1] * watermark.shape[0]) / (img.shape[1] * img.shape[0])))
    wm_width = int(watermark.shape[1] * wm_scale)
    wm_height = int(watermark.shape[0] * wm_scale)
    wm_dim = (wm_width, wm_height)
    resized_wm = cv2.resize(watermark, wm_dim, interpolation=cv2.INTER_AREA)

    ## set watermark location
    h_img, w_img, _ = img.shape
    center_y = int(h_img / 2)
    center_x = int(w_img / 2)
    offset_x = wm_loc % 3 - 1
    offset_y =int(wm_loc / 3) - 1
    h_wm, w_wm, _ = resized_wm.shape
    # set x value
    if offset_x == 0:
        left_x = center_x - int(w_wm / 2)
        right_x = left_x + w_wm
    elif offset_x == -1:
        left_x = WM_MARGIN_x
        right_x = left_x + w_wm
    elif offset_x == 1:
        right_x = w_img - WM_MARGIN_x
        left_x = right_x - w_wm
    else:
        print("# X location Error")
        return(-1)
    # set y value
    if offset_y == 0:
        top_y = center_y - int(h_wm / 2)
        bottom_y = top_y + h_wm
    elif offset_y == -1:
        top_y = WM_MARGIN_Y
        bottom_y = top_y + h_wm
    elif offset_y == 1:
        bottom_y = h_img - WM_MARGIN_Y
        top_y = bottom_y - h_wm
    else:
        print("# Y location Error")
        return(-1)

    # locate ROI area
    roi = img[top_y:bottom_y, left_x:right_x]

    # add watermark to picture ROI area
    if wm_type == 0:
        # METHOD 1: 根据 WM_TRANSPARENCY的值进行叠加
        result = cv2.addWeighted(roi, 1, resized_wm, WM_TRANSPARENCY, 0)
        img[top_y:bottom_y, left_x:right_x] = result
    elif wm_type == 1:
        # METHOD 2: 根据alpha通道的值进行叠加
        # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
        alpha_png = resized_wm[:, :, 3] / 255.0
        alpha_jpg = 1 - alpha_png
        for c in range(0, 3):
            img[top_y:bottom_y, left_x:right_x, c] = ((alpha_jpg * img[top_y:bottom_y, left_x:right_x, c]) + (alpha_png * resized_wm[:, :, c]))
    else:
        print("# invalid wm_type")
        return -1

    # write result to output file
    output_file = os.path.splitext(picfile)[0] + "wm" + os.path.splitext(picfile)[1]
    cv2.imwrite(output_file, img)

    if DEBUG:
        print("# Output file: {}".format(output_file))
        print("#" * 50)

    if DEBUG:
        cv2.imshow("# output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    loc = 4
    wm_type = 0
    print(add_water_mark.__doc__)
    # add_water_mark("images/deer.jpg", "images/watermark.PNG", loc)
    add_water_mark("images/deer.jpg", "images/wm1.png", loc, wm_type)

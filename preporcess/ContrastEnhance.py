import cv2
import numpy as np
def ContrastEnhancement(image):
    #### it work very well

    image=np.array(image,np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)
    # cv2.waitKey(0)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))


    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#    final=np.array(final/255.0,np.float32)
#     cv2.imshow("lab", final)
#     cv2.waitKey(0)
    return final

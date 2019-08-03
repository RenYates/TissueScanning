import numpy as np
import cv2
import glob

def calibrate_camera():

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), ... (6,5,0)
    object_points = np.zeros((7*7,3), np.float32)
    object_points[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images
    obj_points = [] # 3d pt in real world space
    img_points = [] # 2d pts in image plane

    images = glob.glob('Calibration Images/IMG_*.jpg')
    for image in images:
        print("INFO: Searching through images")
        img = cv2.imread(image)
        #img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("INFO: One image found")
            obj_points.append(object_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_points.append(corners2)

            # draw and display the corners
            chess_img = cv2.drawChessboardCorners(img, (7,7), corners2, ret)
            #cv2.imshow('gray', gray)
            #cv2.imshow('img', chess_img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()


    ret, camera_matrix, distortion_coeff, rotation_vec, translation_vec = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    h,  w = chess_img.shape[:2]

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w , h), 1, (w , h))

    print(roi)

    #undistort the image
    dst = cv2.undistort(chess_img, camera_matrix, distortion_coeff, None, new_camera_mtx)

    # crop the image

    x, y, w, h = roi

    new_dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibration_result.png',dst)
    np.save("camera_matrix_orig.npy", camera_matrix)
    np.save("camera_dist_coeff.npy", distortion_coeff)
    np.save("camera_matrix_new.npy", new_camera_mtx)
    print("Camera Matrix:", camera_matrix)
    print("distortion Coeff:", distortion_coeff)
    print("New camera matrix:", new_camera_mtx)
    #return camera_matrix, distortion_coeff, new_camera_mtx

#calibrate_camera()
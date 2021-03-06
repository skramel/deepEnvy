import numpy as np

class calib:
    'Contains the camera calibration parameters'
    
    Npixw = 1280			# Npix_x (pixel)
    Npixh = 1024			# Npix_y (pixel)
    Noffw = 0               # offset_x (pixel)
    Noffh = 0               # offset_y (pixel)
    wpix = 0.014            # pixel size_x (mm)
    hpix = 0.014            # pixel size_y (mm)
    f_eff = 139.27775005         # f_eff - effective focal length
    kr = 0                  # kr - radial distortion coefficient
    kx = 1                  # kx - tangential distortion coefficient
        
#    R = np.zeros( (3,3), dtype=np.float32)
    R = np.array([[-0.73647867, 0.67634937, 0.01227585],
                 [0.10795154, 0.09959541, 0.98915480],
                 [0.66779161, 0.72981661, -0.14636285]])
        
#    T = np.zeros( (3,1), dtype=np.float32)
    T = np.array([-0.59575286, -0.34605043, 1290.93422942])
    

def TsaiProj(calib, p3d):
    """
        Use the calibrated camera parameters to predict the particle position
        projected onto the image plane.
        
        Based on:
        
        R. Tsai, "A versatile camera calibration technique for high-accuracy 3D machine vision metrology using off-the-shelf TV cameras and lenses," in IEEE Journal on Robotics and Automation, vol. 3, no. 4, pp. 323-344, August 1987.
        
        inputs:
            calib   --  calibrated camera parameters
                        Npix_x, Npix_y  # image dimensions (pixel)
                        Noffw, Noffh    # image center
                        wpix, hpix		# pixel size (mm)
                        f_eff           # effective focal length (mm)
            p3D     --  particle coordinates in 3D world coordinates
        
        output:
            p2d     --  projected particle position on image plane (in pixels)
    """

    # Xc = X * R + T
    Xc = np.dot(p3d, calib.R.transpose()) # calib.R' transpose
    Xc = Xc + calib.T

    dummy = calib.f_eff / Xc[:,2]
    Xu = Xc[:,0] * dummy  # undistorted image coordinates
    Yu = Xc[:,1] * dummy
    # calculate radial distorions
    ru2 = Xu**2 + Yu**2
    dummy = 1 + calib.kr*ru2 # k1 distortion parameter
    Xd = Xu * dummy
    Yd = Yu * dummy
    # iterate once
    dummy = 1 + calib.kr*(Xd**2 + Yd**2)
    Xd = Xu * dummy
    Yd = Yu * dummy

    Np = p3d.shape[0]
    p2d = np.zeros( (Np,2) )
    p2d[:,0] = Xd/calib.wpix + calib.Npixw/2 + calib.Noffw
    p2d[:,1] = calib.Npixh/2 - Yd/calib.hpix - calib.Noffh

    return p2d

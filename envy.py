
def TsaiProj(calib, p3d):
    """
        Use the calibrated camera parameters to predict the particle position
        projected onto the image plane.
        
        inputs:
            calib   --  calibrated camera parameters
                        Npix_x, Npix_y  # image dimensions (pixel)
                        wpix, hpix		# pixel size (mm)
                        f_eff           # effective focal length (mm)
            p3D     --  particle coordinates in 3D world coordinates
        
        output:
            p2d     --  projected particle position on image plane (in pixels)
    """

    # Xc = X * R + T
    Xc = p3D * (calib.R) # calib.R' transpose
    Xc(:,1) = Xc(:,1) + calib.T(1)
    Xc(:,2) = Xc(:,2) + calib.T(2)
    Xc(:,3) = Xc(:,3) + calib.T(3)

    # calculate radial distorions
    dummy = calib.f_eff./Xc(:,3)
    Xu = Xc(:,1).*dummy  # undistorted image coordinates
    Yu = Xc(:,2).*dummy
    ru2 = Xu.*Xu + Yu.*Yu
    dummy = 1+calib.k1*ru2 # k1 distortion parameter
    Xd = Xu.*dummy
    Yd = Yu.*dummy
    # iterate once
    dummy = 1 + calib.k1*(Xd.*Xd + Yd.*Yd)
    Xd = Xu.*dummy
    Yd = Yu.*dummy

    Np = size(p3D,1)
    p2d = zeros(Np,2)
    p2d(:,1) = Xd/calib.wpix + calib.Noffw + calib.Npixw/2
    p2d(:,2) = calib.Npixh/2 - calib.Noffh - Yd/calib.hpix

    return p2d

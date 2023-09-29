def estimate_relative_pose_match(matches, cameras, keypoints):
    """
    Estimate essential matrix and relative pose for each camera pair
    """
    #for keypoint in keypoints:
    for match in matches:
        print(
            f"Number of matches between camera {match['camera_i'] + 1} and camera {match['camera_j'] + 1}: {len(match['matches'])}")

        pts_i_flat = np.float32([keypoints[match['camera_i']][m.queryIdx] for m in match['matches']])
        pts_j_flat = np.float32([keypoints[match['camera_j']][m.trainIdx] for m in match['matches']])
        print("Printing pts_i_flat:",pts_i_flat)
        # Slice first two columns to get (x, y) coordinates of keypoints
        pts_i = pts_i_flat[:, :2].reshape(-1, 1, 2)
        pts_j = pts_j_flat[:, :2].reshape(-1, 1, 2)
        print("Printing pts_i_flat:")

        print(pts_i)
        print(f"Shape of pts_j: {pts_j.shape}")

        E, mask = cv2.findEssentialMat(pts_i, pts_j, cameras[match['camera_i']]['matrix'], cv2.RANSAC, 0.5, 10.0)

        num_inliers = np.sum(mask)
        print(f"Number of inliers for camera pair {match['camera_i'] + 1} to {match['camera_j'] + 1}: {num_inliers}")

        _ ,R, t, mask = cv2.recoverPose(E, pts_i, pts_j, cameras[match['camera_i']]['matrix'])

        print(f"Relative pose from camera {match['camera_i']+1} to camera {match['camera_j']+1}:")
        print("Rotation recoverPose:")
        print(R)
        print("Translation recoverPose:")
        print(t)



        # Display keypoints
        img_i = cv2.imread(images[match['camera_i']])
        img_j = cv2.imread(images[match['camera_j']])
        img_joint = np.concatenate((img_i, img_j), axis=1)
        for i in range(len(pts_i)):
            # Draw keypoints on left image
            cv2.circle(img_i, (int(pts_i[i][0][0]), int(pts_i[i][0][1])), 5, (0, 255, 0), -1)
            # Draw keypoints on right image
            cv2.circle(img_j, (int(pts_j[i][0][0]), int(pts_j[i][0][1])), 5, (0, 255, 0), -1)
            # Draw line between keypoints on joint image
            cv2.line(img_joint, (int(pts_i[i][0][0]), int(pts_i[i][0][1])),
                     (int(pts_j[i][0][0]) + img_i.shape[1], int(pts_j[i][0][1])), (0, 255, 0), 1)

        # Display joint image
        cv2.imshow(f"Keypoints for camera {match['camera_i'] + 1} and camera {match['camera_j'] + 1}", img_joint)
        cv2.waitKey(0)

def match_keypoints(keypoints, matcher):
    """
    Match keypoints between pairs of cameras
    """
    matches = []
    for i in range(len(keypoints)):
        for j in range(i+1, len(keypoints)):
            # Match keypoints
            match = matcher.match(keypoints[i], keypoints[j])
            # Sort matches by distance
            match = sorted(match, key=lambda x: x.distance)
            # Check that there are enough matches
            if len(match) < 2:
                print(f"Not enough matches between camera {i+1} and camera {j+1}")
                continue
            matches.append({'matches': match, 'camera_i': i, 'camera_j': j})

            if len(match) == 0:
                print(f"No matches between camera {i + 1} and camera {j + 1}")
                continue


    return matches


    # PnP
    object_points = keypoints1.astype('float32') # 3D
    keypoints2 = keypoints2.astype('float32')
    retval, R_pnp, t_pnp = cv2.solvePnP(object_points, keypoints2, K2, None) # object_points must be 3D

    print(f"solvePnP - Relative pose from camera cam1 to camera cam2:")
    print("Rotation solvePnP: coming soon")
    #print(R_pnp)
    print("Translation solvePnP:coming soon")
    #print(t_pnp)

def triangulate_points(K1, K2, R_recoverPose, t_recoverPose, filtered_keypoints1, filtered_keypoints2):

    num_points = filtered_keypoints1.shape[0]
    points_3d = np.zeros((num_points, 4))
    print(K1)
    print(K2)
    # Compute the projection matrix for the first camera
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Compute the projection matrix for the second camera
    P2 = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose)))

    for i in range(num_points):
        kp1 = np.append(filtered_keypoints1[i], 1)
        kp2 = np.append(filtered_keypoints2[i], 1)

        A = np.zeros((4, 4))

        A[0:2, :] = np.outer(kp1, P1[2, :] - P1[0, :])[:2]
        A[2:4, :] = np.outer(kp2, P2[2, :] - P2[0, :])[:2]

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        X /= X[-1]
        points_3d[i, :] = X

    return points_3d

def reproject_points(points_3d, K2, R_recoverPose, t_recoverPose, filtered_keypoints2):
    num_points = points_3d.shape[0]
    points_homogeneous = np.hstack((points_3d[:, :3], np.ones((num_points, 1))))

    P = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose)))
    points_2d_homogeneous = np.dot(P, points_homogeneous.T).T

    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]

    reprojection_error = np.mean(np.sqrt(np.sum((points_2d - filtered_keypoints2) ** 2, axis=1)))

    return reprojection_error
def reproject_points_solvePnP(points_3d, K2, R_solvePnP, tvec_solvePnP):

    num_points = points_3d.shape[0]
    points_homogeneous = np.hstack((points_3d[:, :3], np.ones((num_points, 1))))

    P = np.dot(K2, np.hstack((R_solvePnP, tvec_solvePnP)))
    points_2d_homogeneous = np.dot(P, points_homogeneous.T).T

    points_2d_solvePnP = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]
    return points_2d_solvePnP
def reproject_points_solvePnPRansac(points_3d, K2, R_solvePnPRansac, tvec_solvePnPRansac):

    num_points = points_3d.shape[0]
    points_homogeneous = np.hstack((points_3d[:, :3], np.ones((num_points, 1))))

    P = np.dot(K2, np.hstack((R_solvePnPRansac, tvec_solvePnPRansac)))
    points_2d_homogeneous = np.dot(P, points_homogeneous.T).T

    points_2d_solvePnPRansac = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]
    return points_2d_solvePnPRansac
def reproject_points_solvePnPRefineLM(points_3d, K2, R_solvePnPRefineLM, tvec_refined):
    num_points = points_3d.shape[0]
    points_homogeneous = np.hstack((points_3d[:, :3], np.ones((num_points, 1))))

    P = np.dot(K2, np.hstack((R_solvePnPRefineLM, tvec_refined)))
    points_2d_homogeneous = np.dot(P, points_homogeneous.T).T

    points_2d_solvePnPRefineLM = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]

    return points_2d_solvePnPRefineLM

    # reprojection_error = reproject_points(triangulated_points, K2, R_recoverPose, t_recoverPose,filtered_keypoints2)
    # #reprojection_error = np.mean(np.sqrt(np.sum((reprojected_keypoints2 - filtered_keypoints2) ** 2, axis=1)))
    # print('Print reprojection error recoverPose:',reprojection_error)
    #
    # reprojected_keypoints2_solvePnP = reproject_points_solvePnP(triangulated_points, K2, R_solvePnP, tvec_solvePnP)
    # reprojection_error_solvePnP = np.mean(np.sqrt(np.sum((reprojected_keypoints2_solvePnP - filtered_keypoints2) ** 2, axis=1)))
    # print('Print reprojection error solvePnP:', reprojection_error_solvePnP)
    #
    # reprojected_keypoints2_solvePnPRansac = reproject_points_solvePnPRansac(triangulated_points, K2, R_solvePnPRansac, tvec_solvePnPRansac)
    # reprojection_error_solvePnPRansac = np.mean(np.sqrt(np.sum((reprojected_keypoints2_solvePnPRansac - filtered_keypoints2) ** 2, axis=1)))
    # print('Print reprojection error solvePnPRansac:', reprojection_error_solvePnPRansac)
    #
    # reprojected_keypoints2_solvePnPRefineLM = reproject_points_solvePnPRefineLM(triangulated_points, K2, R_solvePnPRefineLM, tvec_refined)
    # reprojection_error_solvePnPRefineLM = np.mean(np.sqrt(np.sum((reprojected_keypoints2_solvePnPRefineLM - filtered_keypoints2) ** 2, axis=1)))
    # print('Print reprojection error solvePnPRefineLM:', reprojection_error_solvePnPRefineLM)

  # RMS
    num_keypoints = filtered_keypoints1.shape[0]
    print('Number of keypoints:',num_keypoints)

    rms_error = np.sqrt(reprojection_error / num_keypoints)
    rms_error_solvePnP = np.sqrt(reprojection_error_solvePnP / num_keypoints)
    rms_error_solvePnPRefineLM = np.sqrt(reprojection_error_solvePnPRefineLM / num_keypoints)

    print('Print RMS error recoverPose: ' ,rms_error)
    print('Print RMS error solvePnP: ', rms_error_solvePnP)
    print('Print RMS error solvePnPRefineLM: ', rms_error_solvePnPRefineLM)

    # UNDISTORT
    keypoints1 = cv2.undistortPoints(np.expand_dims(keypoints1, axis=1), cameraMatrix=camera_matrix, distCoeffs=None)
    keypoints2 = cv2.undistortPoints(np.expand_dims(keypoints2, axis=1), cameraMatrix=camera_matrix, distCoeffs=None)

    # Estimate the essential matrix
    E, mask = cv2.findEssentialMat(keypoints1, keypoints2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1)

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, solver_type, linear_solver):
        super().__init__()
        solver_block = solver_type(linear_solver())
        solver = g2o.OptimizationAlgorithmLevenberg(solver_block)
        super().set_algorithm(solver)
        self.set_verbose(True)  # Add this line to enable verbosity

    def optimize(self, max_iterations):
        self.initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, fx, fy, cx, cy, baseline, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(fx, fy, cx, cy, baseline)
        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        self.add_vertex(v_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        self.add_vertex(v_p)

    def add_observation_edge(self, point_id, pose_id, measurement, information, robust_kernel_threshold):
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id + len(self.vertices())))
        edge.set_vertex(1, self.vertex(pose_id))
        edge.set_measurement(np.array(measurement).reshape(2, 1))
        edge.set_information(information)

        if robust_kernel_threshold is not None:
            edge.set_robust_kernel(g2o.RobustKernelHuber(robust_kernel_threshold))

        self.add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id).estimate()

    def get_point(self, point_id):
        vertex = self.vertex(point_id * 2 + 1)
        if isinstance(vertex, g2o.VertexPointXYZ):
            return np.array(vertex.estimate())
        else:
            print(f"Warning: Point with ID {point_id} was not retrieved correctly!")
            return None

def perform_bundle_adjustment_with_class(points, cam_matrices, rotations, translations, keypoints_list,
                                          num_iterations,
                                          robust_kernel_threshold,
                                          information_matrices_list,
                                          solver_type,
                                          linear_solver):
    # Create an instance of the BundleAdjustment class
    ba = BundleAdjustment(solver_type=solver_type, linear_solver=linear_solver)

    # Add camera poses
    for i, (R, t) in enumerate(zip(rotations, translations)):
        pose = g2o.Isometry3d(R, t)
        fx, fy, cx, cy = cam_matrices[i][0, 0], cam_matrices[i][1, 1], cam_matrices[i][0, 2], cam_matrices[i][1, 2]
        baseline = 0  # Modify as needed
        fixed = (i == 0)
        ba.add_pose(pose_id=i, pose=pose, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline, fixed=fixed)

    # Add 3D points and their observations
    for point_id, point in enumerate(points):
        ba.add_point(point_id=point_id, point=point)

        # Add edges for observations from both cameras
        for cam_id in [0, 1]:
            measurement = keypoints_list[cam_id][point_id]
            #information = information_matrices_list[cam_id][point_id]
            information = np.identity(2)
            ba.add_observation_edge(point_id=point_id, pose_id=cam_id, measurement=measurement,
                        information=information, robust_kernel_threshold=robust_kernel_threshold)

    # Optimize
    ba.optimize(max_iterations=num_iterations)

    # Extract the optimized camera poses and points
    optimized_poses = [ba.get_pose(pose_id=i) for i in range(len(rotations))]
    #optimized_points = [np.array(ba.get_point(point_id=i)).reshape(-1) for i in range(len(points)) if ba.get_point(point_id=i) is not None]

    optimized_points = []
    missing_point_ids = []
    for i in range(len(points)):
        point = ba.get_point(point_id=i)
        if point is not None:
            optimized_points.append(np.array(point).reshape(-1))
        else:
            missing_point_ids.append(i)


    return optimized_points, optimized_poses, missing_point_ids
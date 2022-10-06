import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from scipy.spatial import KDTree

from pydrake.all import RigidTransform

# Store X_lst_target as global for testing all the functions
# yapf: disable
X_lst_target = np.array([  # noqa
    [[-0.20928706, -0.97758728, 0.02284888, 0.0225679],
     [0.96642625, -0.21034659, -0.14756241, 0.02382296],
     [0.14906132, -0.00880115, 0.98878878, 0.08232251]],
    [[-0.73116648, 0.15981375, 0.66321576, -0.00774401],
     [-0.5801478, -0.65714514, -0.48123673, 0.03270167],
     [0.35892078, -0.73662734, 0.57319808, 0.14404616]],
    [[-0.35058022, -0.93627131, 0.02212612, 0.01865798],
     [0.93135369, -0.35102537, -0.09675485, 0.03471024],
     [0.09835561, -0.01331309, 0.99506229, 0.10688465]],
    [[-0.84367496, 0.52562952, -0.10920669, -0.01526701],
     [-0.46827883, -0.82000023, -0.32911178, 0.04316992],
     [-0.2625404, -0.22652419, 0.93795484, 0.04541413]]])
# yapf: enable

test_indices = [10137, 21584, 7259, 32081]


class TestGraspCandidate(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(10.)
    def test_darboux_frame(self):
        """Test compute_darboux_frame"""
        pcd = self.notebook_locals["pcd"]
        kdtree = KDTree(pcd.xyzs().T)
        f = self.notebook_locals["compute_darboux_frame"]

        X_lst_eval = []

        for i in range(4):
            index = test_indices[i]
            RT = f(index, pcd, kdtree)
            X_lst_eval.append(RT.GetAsMatrix34())

        X_lst_eval = np.asarray(X_lst_eval)

        self.assertLessEqual(np.linalg.norm(X_lst_target - X_lst_eval), 0.02,
                             "The Darboux frame is not correct")

        index = 5
        RT = f(index, pcd, kdtree)

        X_lst_order_eval = RT.GetAsMatrix34()

        # yapf: disable
        X_lst_order_target = np.array([  # noqa
        [0.03664805, -0.88053584, -0.4725607, 0.00884411],
        [0.9375121, 0.19402823, -0.28883234, -0.00240758],
        [0.34601733, -0.43244621, 0.83262372, 0.19118714]])
        # yapf: enable

        self.assertLessEqual(
            np.linalg.norm(X_lst_order_eval - X_lst_order_target), 1e-4,
            "Did you forget to sort the eigenvalues, "
            "or handle improper rotations?")

    @weight(4)
    @timeout_decorator.timeout(10.)
    def test_minimum_distance(self):
        """Test find_minimum_distance"""
        pcd = self.notebook_locals["pcd"]
        f = self.notebook_locals["find_minimum_distance"]

        # The following should return nan
        for i in [0, 2]:
            dist, X_new = f(pcd, RigidTransform(X_lst_target[i]))
            self.assertTrue(
                np.isnan(dist), "There is no value of y that results in "
                "no collision in the grid, but dist is not nan")
            self.assertTrue(
                isinstance(X_new, type(None)),
                "There is no value of y that results in "
                "no collision in the grid, but X_WGnew is"
                "not None.")

        # yapf: disable
        dist_new_target = np.array([  # noqa
            0.00357997,
            0.00080692])

        X_new_target = np.array([  # noqa
        [[-0.73116648, 0.15981375, 0.66321576, -0.0157347],
         [-0.5801478, -0.65714514, -0.48123673, 0.06555893],
         [0.35892078, -0.73662734, 0.57319808, 0.18087752]],
        [[-0.84367496, 0.52562952, -0.10920669, -0.03570816],
         [-0.46827883, -0.82000023, -0.32911178, 0.07505881],
         [-0.2625404, -0.22652419, 0.93795484, 0.05422341]]])
        # yapf: enable

        dist_new_eval = []
        X_new_eval = []
        # The following should return numbers.
        for i in [1, 3]:
            dist, X_new = f(pcd, RigidTransform(X_lst_target[i]))
            self.assertTrue(
                not np.isnan(dist),
                "There is a valid value of y that results in "
                "no collision in the grid, but dist is nan")
            self.assertTrue(
                not isinstance(X_new, type(None)),
                "There is a valid value of y that results in no "
                "collision in the grid, but X_WGnew is None.")
            dist_new_eval.append(dist)
            X_new_eval.append(X_new.GetAsMatrix34())

        dist_new_eval = np.array(dist_new_eval)
        X_new_eval = np.array(X_new_eval)

        self.assertLessEqual(np.linalg.norm(dist_new_target - dist_new_eval),
                             1e-5, "The returned distance is not correct.")
        self.assertLessEqual(np.linalg.norm(X_new_target - X_new_eval), 1e-4,
                             "The returned transform is not correct.")

    @weight(4)
    @timeout_decorator.timeout(60.)
    def test_candidate_grasps(self):
        """Test compute_candidate_grasps"""
        pcd = self.notebook_locals["pcd_downsampled"]
        compute_candidate_grasps = self.notebook_locals[
            "compute_candidate_grasps"]
        find_minimum_distance = self.notebook_locals["find_minimum_distance"]
        check_collision = self.notebook_locals["check_collision"]
        check_nonempty = self.notebook_locals["check_nonempty"]

        grasp_candidates = compute_candidate_grasps(pcd,
                                                    candidate_num=3,
                                                    random_seed=5)

        self.assertTrue(
            len(grasp_candidates) == 3,
            "Length of returned array is not correct.")

        for X_WP in grasp_candidates:
            distance, X_WP_new = find_minimum_distance(pcd, X_WP)
            self.assertLessEqual(
                distance,
                1e-2), "The returned grasp candidates are not minimum distance"
            self.assertTrue(check_collision(
                pcd, X_WP)), "The returned grasp candidates have collisions"
            self.assertTrue(check_nonempty(
                pcd, X_WP)), "The returned grasp candidates are not empty"

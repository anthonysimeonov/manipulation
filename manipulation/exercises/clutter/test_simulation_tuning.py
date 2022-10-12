import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import os
from pydrake.all import (RigidTransform, RollPitchYaw, BodyIndex)


class TestSimulationTuning(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @timeout_decorator.timeout(10.)
    @weight(3)
    def test_on_slope(self):
        """Test test_on_slope"""
        # part a. -- check if object is on slope
        # set_hyp = self.notebook_locals["set_hyperparameter_552"]
        # make_simulation = self.notebook_locals["make_simulation"]
        # simulator, diagram = make_simulation(*set_hyp())
        simulator = self.notebook_locals["simulator552a"]
        diagram = self.notebook_locals["diagram552a"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]
        print(f'Box 1: {box1_pos}')
        print(f'Box 2: {box2_pos}')

        # box1_frame = plant.GetBodyByName('box').body_frame()
        # box2_frame = plant.GetBodyByName('box_2').body_frame()

        # box1_pose = box1_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # box2_pose = box2_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # print(f'Box 1 pose: {box1_pose}')
        # print(f'Box 2 pose: {box1_pose}')

        box_pos_range = np.array([
            [0.0, 1.0], 
            [-0.1, 0.1], 
            [0.03, 0.07], 
        ])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = box_pos_range[i, 0] < box1_pos[i] < box_pos_range[i, 1]
            in_range_2 = box_pos_range[i, 0] < box2_pos[i] < box_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(in_range_all.all(), 'Final box positions not in correct range!')


    @timeout_decorator.timeout(10.)
    @weight(3)
    def test_make_simulation(self):
        """Test test_make_simulation"""
        # part c. --  check if objects are in desired final range
        simulator = self.notebook_locals["simulator552c"]
        diagram = self.notebook_locals["diagram552c"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]
        print(f'Box 1: {box1_pos}')
        print(f'Box 2: {box2_pos}')

        # box1_frame = plant.GetBodyByName('box').body_frame()
        # box2_frame = plant.GetBodyByName('box_2').body_frame()

        # box1_pose = box1_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # box2_pose = box2_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # print(f'Box 1 pose: {box1_pose}')
        # print(f'Box 2 pose: {box2_pose}')

        box_pos_range = np.array([
            [0.0, 1.0], 
            [-0.1, 0.1], 
            [0.03, 0.07], 
        ])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = box_pos_range[i, 0] < box1_pos[i] < box_pos_range[i, 1]
            in_range_2 = box_pos_range[i, 0] < box2_pos[i] < box_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(in_range_all.all(), 'Final box positions not in correct range!')

    @timeout_decorator.timeout(10.)
    @weight(3)
    def test_make_stacking_simulation(self):
        """Test test_make_stacking_simulation"""
        # 5.3.3 -- check if objects are stacked + at desired range
        # set_hyp = self.notebook_locals["set_hyperparameter_552"]
        # make_simulation = self.notebook_locals["make_simulation"]
        # simulator, diagram = make_simulation(*set_hyp())
        simulator = self.notebook_locals["simulator553"]
        diagram = self.notebook_locals["diagram553"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]
        print(f'Box 1: {box1_pos}')
        print(f'Box 2: {box2_pos}')

        # box1_frame = plant.GetBodyByName('box').body_frame()
        # box2_frame = plant.GetBodyByName('box_2').body_frame()

        # box1_pose = box1_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # box2_pose = box2_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # print(f'Box 1 pose: {box1_pose}')
        # print(f'Box 2 pose: {box2_pose}')

        # # box1_frame = plant.get_body(1).body_frame()
        # # box2_frame = plant.get_body(2).body_frame()
        # box1_frame = plant.GetBodyByName('box').body_frame()
        # box2_frame = plant.GetBodyByName('box_2').body_frame()

        # box1_pose = (box1_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()[:3,3])
        # box2_pose = (box2_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()[:3,3])
        # body_frame_velocity = box1_frame.CalcSpatialVelocityInWorld(plant_context).translational()
        # body_frame_velocity2 = box2_frame.CalcSpatialVelocityInWorld(plant_context).translational()
        # print(f"box 1 pose: {box1_pose} box 2 pose: {box2_pose}")
        # print(f"box 1 velocity: {body_frame_velocity} box 2 velocity: {body_frame_velocity2}")
        
        box1_pos_range = np.array([
            [0.2, 0.3], 
            [-0.025, 0.025], 
            [0.025, 0.0375]
            ])
        box2_pos_range = np.array([
            [0.2, 0.3], 
            [-0.025, 0.025], 
            [0.0475, 0.06] 
            ])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = box1_pos_range[i, 0] < box1_pos[i] < box1_pos_range[i, 1]
            in_range_2 = box2_pos_range[i, 0] < box2_pos[i] < box2_pos_range[i, 1]
            # print(f'Box 1: {box1_pos[i]}, Box 2: {box2_pos[i]}')
            # print(f'In 1: {in_range_1}, In 2: {in_range_2}')
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(in_range_all.all(), 'Final box positions not in correct range!')

    @weight(1)
    def test_matching_coll_shape(self):
        """Test test_matching_coll_shape"""
        # set_hyp = self.notebook_locals["set_hyperparameter_551"]
        # make_teleop_simulation = self.notebook_locals["make_teleop_simulation"]
        # plant, diagram = make_teleop_simulation(*set_hyp(), interactive=False)
        plant = self.notebook_locals["plant551"]
        diagram = self.notebook_locals["diagram551"]

        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        contact_results = plant.get_contact_results_output_port().Eval(plant_context)

        # we can use these positions to test corner overlap
        box_q_list = np.array([
            [0., 0.15, 0.],
            [0., 0.2, 0.],
            [0., 0.2, 0.],
            [0., 0.13, 0.]
            ])

        box2_q_list = np.array([
            [-0.049, 0.198, 0.],
            [-0.049, 0.155, 0.],
            [0.046, 0.168, 0.],
            [0.046, 0.168, 0.]
            ])
        
        n_contacts_list = []
        for i in range(4):
            plant.SetPositions(plant_context, np.concatenate([box_q_list[i], box2_q_list[i]]))  # TODO: how to set the joint positions properly?

            contact_results = plant.get_contact_results_output_port().Eval(plant_context)
            n_contacts = contact_results.num_point_pair_contacts()

            n_contacts_list.append(n_contacts)

        print(f'List of contacts: {n_contacts_list}')

        self.assertTrue((np.array(n_contacts) > 0).all(), 'Objects not contacting!')

    @weight(1)
    def test_force_discontinuity(self):
        """Test test_force_discontinuity"""
        plant = self.notebook_locals["plant551"]
        diagram = self.notebook_locals["diagram551"]
        f = self.notebook_locals["set_block_2d_poses"]

        box1_pos1, box2_pos1, box1_pos2, box2_pos2 = f()

        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        plant.SetPositions(plant_context, np.concatenate([box1_pos1 + box2_pos1])) 

        contact_port = plant.get_contact_results_output_port()
        contact_results = plant.get_contact_results_output_port().Eval(plant_context)
        n_contacts = contact_results.num_point_pair_contacts()
        info = contact_results.point_pair_contact_info(0)
        normal_1 = info.contact_force()
        point_1 = info.contact_point()
        print(f'Normal 1: {normal_1}\nContact point 1: {point_1}\n\n')

        plant.SetPositions(plant_context, np.concatenate([box1_pos2 + box2_pos2]))

        contact_port = plant.get_contact_results_output_port()
        contact_results = plant.get_contact_results_output_port().Eval(plant_context)
        n_contacts = contact_results.num_point_pair_contacts()
        info = contact_results.point_pair_contact_info(0)
        normal_2 = info.contact_force()
        point_2 = info.contact_point()
        print(f'Normal 2: {normal_2}\nContact point 2: {point_2}\n\n')

        contact_pt_dist = np.linalg.norm(point_1 - point_2)
        force_angle = np.arccos(np.dot(normal_1, normal_2))
        print(f'Contact point dist: {contact_pt_dist:5f}\nAngle between forces: {force_angle:5f}\n\n')

        close_pts = contact_pt_dist < 0.01
        self.assertTrue(close_pts, 'Contact points not close enough, discontinuity not detected') 

        large_angle = force_angle > np.deg2rad(60)
        self.assertTrue(large_angle, 'Angle between force vectors too small, discontinuity not detected')

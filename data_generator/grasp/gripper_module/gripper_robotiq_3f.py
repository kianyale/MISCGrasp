import time
from pathlib import Path
import os
import pybullet
import sys

sys.path.append("/")

from src.gd.grasp import Label
from src.gd.perception import *
from src.gd.io import *
from src.gd.utils import btsim, workspace_lines
from src.gd.utils.transform import Rotation, Transform
from .gripper_base import GripperBase


class GripperRobotiq3F(GripperBase):
    """
    Initialization of Barrett-Hand-2f gripper
    """
    name = 'robotiq_3f'

    def __init__(self, world, gripper_size=1.):
        self.world = world
        self._gripper_size = gripper_size
        self.urdf_path = Path("/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/gripper/robotiq_3f/combined_mount_robotiq_3f"
                              ".urdf")
        # self.urdf_path = Path("/data/Group/Grasp/adagrasp/.tmp_combined_barrett_hand_2f_1.0000_0.4144073594_1712669888.8483331203.urdf")
        self.max_opening_width = 0.16 * self._gripper_size
        # 'left' and 'right' are defined relative to the x-axis of the gripper's reference frame.
        self.finger_open_distance_right = 0.08 * self._gripper_size  # single-finger side
        self.finger_open_distance_left = 0.08 * self._gripper_size  # two-finger side
        self.half_gap_2f = 0.0375 * self._gripper_size  # Distance from one of the two fingers to the y-axis (half of
        # the gap between the two fingers)

        self.finger_depth_init = 0.112 * self._gripper_size
        self.finger_depth = 0.107 * self._gripper_size  # 0.133 0.125
        self.T_body_tcp_first = Transform(Rotation.identity(), [0.0, 0.0, self.finger_depth]) * \
                                Transform(Rotation.from_euler('x', 180, degrees=True), [0., 0., 0.]) * \
                                Transform(Rotation.from_euler('z', -np.pi / 2), [0., 0., 0.])
        self.T_tcp_body_first = self.T_body_tcp_first.inverse()
        self.T_tcp_trlink = Transform(Rotation.from_matrix(np.array([[0., 0., 1.],
                                                                     [1., 0., 0.],
                                                                     [0., 1., 0.]])),
                                      np.array([0., 0., -0.051]) * self._gripper_size)
        # self.T_tcp_trlink = Transform(Rotation.from_matrix(np.array([[0., 0., 0.],
        #                                                              [0., 0., 0.],
        #                                                              [0., 0., 0.]])),
        #                               np.array([0., 0., 0.]) * self._gripper_size)

        self._gripper_parent_index = 4

        # define driver joint; the follower joints need to satisfy constraints when grasping
        finger1_joint_ids = [1, 2, 3]
        finger2_joint_ids = [5, 6, 7]
        finger3_joint_ids = [9, 10, 11]
        self._finger_joint_ids = finger1_joint_ids + finger2_joint_ids + finger3_joint_ids
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]

        self._palm_joint_ids = [0, 4]  # the joints that link the palm and fingers. not moved in grasping

        self._joint_lower = 0.1
        self._joint_upper = 0.45
        # define force and speed (movement of mount)
        self._force = 800
        self._speed = 0.1  # 0.005
        # define force and speed (grasping)
        self._grasp_force = 50
        self._grasp_speed = 2
        self.inherent_link_len = 0.1 * self._gripper_size

    def reset(self, T_world_tcp):
        """
        Parameters:
            T_world_tcp(Transform): the pose of the gripper in the world
        Remarks:
            Once the gripper is placed, T_world_body is determined and won't be changed.
        """
        self.T_world_body = T_world_tcp * self.T_tcp_body_first

        self.body = self.world.load_urdf(self.urdf_path, self.T_world_body, self._gripper_size)
        self.body.configure(self._gripper_parent_index + 1)
        self.fix_joints(range(self.world.p.getNumJoints(self.body.uid)))
        self.move(1.0)

        self.joint1 = self.body.joints["finger_1_joint_1"]
        self.joint2 = self.body.joints["finger_2_joint_1"]
        self.joint3 = self.body.joints["finger_middle_joint_1"]

        self.traced_link = self.body.links["palm"]

    def set_tcp(self, T_world_tcp):
        """
        Raises:
            TBD
        """
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, abort_on_contact=True, rotation_angle=0.):
        """
        Parameters:
            target(Transform): it is in the world frame, i.e. T_world_target, which needs to transfer into the body frame first
            rotation_angle(float): rotation in z axis of the tcp frame \in [0, 2 * \pi]. For 2-finger gripper, angle=0 --> parallel to x-axis
        Remarks:
             The target of "setJointMotorControlArray" function is based on the fingertip center in the closed state of
             the gripper with the same orientation as the body frame. That is, when inputting a T_body_target, it will
             have the fingertip center moving into the target position not the tcp. So, when calculating the T_body_target,
             we input a target slightly lower than the position of T_body_target by one finger in length(0.05).
        """
        joint_ids = [0, 1, 2, 3]
        T_world_trlink = self.traced_link.get_pose()
        T_world_tcp = T_world_trlink * self.T_tcp_trlink.inverse()
        T_body_tcp = self.T_world_body.inverse() * T_world_tcp
        T_body_target = self.T_world_body.inverse() * target

        diff = T_body_target.translation - T_body_tcp.translation
        # target_virtual: T_body_(target_tcp)
        target_virtual = T_body_target * Transform(Rotation.identity(), [0., 0., self.finger_depth])
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / (n_steps + 1e-5)
        dur_step = np.linalg.norm(dist_step) / self._speed

        self.world.p.setJointMotorControlArray(
            self.body.uid,
            joint_ids,
            self.world.p.POSITION_CONTROL,
            targetPositions=[*target_virtual.translation, rotation_angle % (2 * np.pi)],
            forces=[self._force] * len(joint_ids),
            positionGains=[self._speed] * len(joint_ids)
        )

        for _ in range(n_steps):
            for _ in range(int(dur_step / self.world.dt)):
                self.step_constraints()
                self.world.step()
            if abort_on_contact and self.detect_contact():
                break
        self.fix_joints(joint_ids)

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, open_scale):
        if open_scale == 0.:
            self.close()
        else:
            self.open(open_scale)

    def read(self):
        ratio = ((
                             self.joint1.get_position() + self.joint2.get_position() + self.joint3.get_position() - 3 * self._joint_lower) /
                 ((self._joint_upper - self._joint_lower) * 3))
        return 1 - ratio

    def open(self, open_scale):
        # recalculate scale because larger joint position corresponds to smaller open width
        open_pos = open_scale * self._joint_lower + (
                1 - open_scale) * self._joint_upper
        self.world.p.setJointMotorControl2(
            self.body.uid,
            self._driver_joint_id + self._gripper_parent_index + 1,
            self.world.p.POSITION_CONTROL,
            targetPosition=open_pos,
            force=self._grasp_force,
            maxVelocity=1
        )
        for i in range(int(2. / self.world.dt)):
            pos = self.step_constraints()
            if np.abs(open_pos - pos) < 1e-5:
                break
            self.world.step()

    def close(self):
        open_pos = self._joint_upper
        self.world.p.setJointMotorControl2(
            self.body.uid,
            self._driver_joint_id + self._gripper_parent_index + 1,
            self.world.p.POSITION_CONTROL,
            targetPosition=open_pos,
            force=self._grasp_force,
            maxVelocity=self._grasp_speed
        )
        for i in range(int(1. / self.world.dt)):
            pos = self.step_constraints()
            if pos > self._joint_upper:
                break
            self.world.step()

    def step_constraints(self):
        # fix 2 fingers in 0 position
        self.world.p.setJointMotorControlArray(
            self.body.uid,
            [i + self._gripper_parent_index + 1 for i in self._palm_joint_ids],
            self.world.p.POSITION_CONTROL,
            targetPositions=[0.0] * 2,
            forces=[self._grasp_force] * 2,
            positionGains=[1.6] * 2
        )
        pos = self.world.p.getJointState(self.body.uid, self._driver_joint_id + self._gripper_parent_index + 1)[0]
        self.world.p.setJointMotorControlArray(
            self.body.uid,
            [i + self._gripper_parent_index + 1 for i in self._follower_joint_ids],
            self.world.p.POSITION_CONTROL,
            targetPositions=[pos, pos - 0.5, pos, pos, pos - 0.5, pos, pos, pos - 0.5],
            forces=[self._grasp_force] * len(self._follower_joint_ids),
            positionGains=[1.2] * len(self._follower_joint_ids)
        )
        return pos

    def fix_joints(self, joint_ids):
        current_states = np.array(
            [self.world.p.getJointState(self.body.uid, joint_id)[0] for joint_id in joint_ids])
        self.world.p.setJointMotorControlArray(
            self.body.uid,
            joint_ids,
            self.world.p.POSITION_CONTROL,
            targetPositions=current_states,
            forces=[self._force] * len(joint_ids),
            positionGains=[self._speed] * len(joint_ids)
        )


if __name__ == '__main__':
    import pybullet as p
    from pybullet_utils import bullet_client


    def check_success(world, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1
        return res


    gui = True
    world = btsim.BtWorld(gui)
    gripper = GripperRobotiq3F(world)
    print(-gripper.finger_depth)
    T_world_grasp = Transform(Rotation.from_euler('XYZ', [-np.pi, np.pi / 2, 0.]),
                              [0., -gripper.finger_open_distance_right, 0.0375])
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.112])
    T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
    approach = T_world_grasp.rotation.as_matrix()[:, 2]
    angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
    if angle > np.pi / 3.0:
        # side grasp, lift the object after establishing a grasp
        T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.2])
        T_world_retreat = T_grasp_pregrasp_world * T_world_grasp  # left multiplication: basing on the world frame
    else:
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.2])
        T_world_retreat = T_world_grasp * T_grasp_retreat

    T_temp = Transform(Rotation.identity(), [0., 0., 0.])
    gripper.reset(T_temp)
    T_world_trlink = gripper.body.links['palm'].get_pose()  # at this moment, world = tcp
    print(T_world_trlink.as_matrix())
    time.sleep(20)

    allow_contact = False
    remove = True
    T_tcp_trlink = gripper.T_tcp_trlink

    gripper.reset(T_world_pregrasp)
    T_world_trlink = gripper.body.links['palm'].get_pose()
    print(T_world_trlink)
    print((T_world_trlink * T_tcp_trlink.inverse()).as_matrix(), '\n')
    # gripper.move(0.)
    time.sleep(30)

    if gripper.detect_contact():
        result = Label.FAILURE, gripper.max_opening_width
    else:
        gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
        T_world_trlink = gripper.body.links['palm'].get_pose()
        print((T_world_trlink * T_tcp_trlink.inverse()).as_matrix(), '\n')
        time.sleep(2)

        if gripper.detect_contact() and not allow_contact:
            result = Label.FAILURE, gripper.max_opening_width
        else:
            gripper.move(0.)
            print(gripper.read())
            # print(gripper.link1.get_pose().translation)
            # print(gripper.link2.get_pose().translation)
            # print(gripper.joint1.get_position())
            time.sleep(20)

            gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
            T_world_trlink = gripper.body.links['palm'].get_pose()
            print((T_world_trlink * T_tcp_trlink.inverse()).as_matrix(), '\n')
            time.sleep(2)

            if check_success(world, gripper):
                result = Label.SUCCESS, gripper.read()
                if remove:
                    contacts = world.get_contacts(gripper.body)
                    world.remove_body(contacts[0].bodyB)
            else:
                result = Label.FAILURE, gripper.max_opening_width
    gripper.move(0.25)
    print(gripper.read())
    # print(gripper.link1.get_pose().translation)
    # print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    gripper.move(0.5)
    print(gripper.read())
    # print(gripper.link1.get_pose().translation)
    # print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    gripper.move(0.75)
    print(gripper.read())
    # print(gripper.link1.get_pose().translation)
    # print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    gripper.move(1.)
    print(gripper.read())
    # print(gripper.link1.get_pose().translation)
    # print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    # gripper.move(0.)
    print(result)
    print("retreat:", '\n', T_world_retreat.as_matrix())
    # gripper.close()
    print(gripper.read())
    # print(gripper.link1.get_pose().translation)
    # print(gripper.link2.get_pose().translation)
    # gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False)
    gripper.move(0.)
    print(gripper.joint1.get_position())
    print(gripper.joint2.get_position())
    print(gripper.joint3.get_position())
    # T_world_origin = T_world_retreat * Transform(Rotation.identity(), [-0.079, 0.15, 0.17])
    # gripper.move_tcp_xyz(T_world_origin)
    time.sleep(30)

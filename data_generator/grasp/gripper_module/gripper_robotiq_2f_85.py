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


class GripperRobotiq2F85(GripperBase):
    """
    Initialization of robotiq-2f-85 gripper
    """
    name = 'robotiq_2f_85'
    def __init__(self, world, gripper_size=1.):
        self.world = world
        self._gripper_size = gripper_size
        self.urdf_path = Path("/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/gripper/robotiq_2f_85/combined_mount_robotiq_2f_85.urdf")
        self.max_opening_width = 0.085 * self._gripper_size
        self.finger_depth = 0.073 * self._gripper_size
        self.finger_depth_init = 0.059 * self._gripper_size
        self.T_body_tcp_first = Transform(Rotation.identity(), [0.0, 0.0, self.finger_depth]) * \
                                Transform(Rotation.from_euler('x', 180, degrees=True), [0., 0., 0.]) * \
                                Transform(Rotation.from_euler('z', -np.pi / 2), [0., 0., 0.])
        self.T_tcp_body_first = self.T_body_tcp_first.inverse()
        self.T_tcp_trlink = Transform(Rotation.from_matrix(np.array([[1., 0., 0.],
                                                                     [0., 1., 0.],
                                                                     [0., 0., 1.]])),
                                      np.array([0., 0., -0.0606]) * self._gripper_size)

        self._gripper_parent_index = 4
        # define driver joint; the follower joints need to satisfy constraints when grasping
        self._driver_joint_id = 0
        self._driver_joint_lower = 0
        self._driver_joint_upper = 0.8  # NOTE
        self._follower_joint_ids = [5, 2, 7, 4, 9]
        self._follower_joint_sign = [1, -1, -1, 1, 1]
        # define force and speed (movement of mount)
        self._force = 800
        self._speed = 0.025  # 0.005
        # define force and speed (grasping)
        self._grasp_force = 50  # TODO
        self._grasp_speed = 1
        self.inherent_link_len = 0.0834 * self._gripper_size

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

        self.link1 = self.body.links["left_inner_finger"]
        self.link2 = self.body.links["right_inner_finger"]
        self.traced_link = self.body.links["robotiq_arg2f_base_link"]

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
        diff = self.link1.get_pose().translation - self.link2.get_pose().translation
        width = np.linalg.norm(diff) - self.inherent_link_len
        return width / self.max_opening_width

    def open(self, open_scale):
        # recalculate scale because larger joint position corresponds to smaller open width
        target_pos = open_scale * self._driver_joint_lower + (1 - open_scale) * self._driver_joint_upper

        self.world.p.setJointMotorControl2(
            self.body.uid,
            self._driver_joint_id + self._gripper_parent_index + 1,
            self.world.p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=self._grasp_force
        )
        for i in range(int(0.5 / self.world.dt)):
            driver_pos = self.step_constraints()
            if np.abs(driver_pos - target_pos) < 1e-5:
                break
            self.world.step()

    def close(self):
        self.world.p.setJointMotorControl2(
            self.body.uid,
            self._driver_joint_id + self._gripper_parent_index + 1,
            self.world.p.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,  # afterward pay attention to such symbols
            force=self._grasp_force,
        )

        targets = self._driver_joint_upper * np.array(self._follower_joint_sign)

        # import ipdb ;  ipdb.set_trace()
        for i in range(int(1. / self.world.dt)):
            pos = self.step_constraints()
            if pos > self._driver_joint_upper:
                break
            self.world.step()

    def step_constraints(self):
        pos = self.world.p.getJointState(self.body.uid, self._driver_joint_id + self._gripper_parent_index + 1)[0]
        targets = pos * np.array(self._follower_joint_sign)
        self.world.p.setJointMotorControlArray(
            self.body.uid,
            [(joint_id + self._gripper_parent_index + 1) for joint_id in self._follower_joint_ids],
            self.world.p.POSITION_CONTROL,
            targetPositions=targets,
            forces=[self._grasp_force] * len(self._follower_joint_ids),
            positionGains=[1] * len(self._follower_joint_ids)
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
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


    gui = True
    world = btsim.BtWorld(gui)
    gripper = GripperRobotiq2F85(world)
    T_world_grasp = Transform(Rotation.from_euler('XYZ', [-np.pi, np.pi / 2, 0.]), [0., -0.085 / 2, 0.])
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -gripper.finger_depth_init])
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

    allow_contact = False
    remove = True
    T_tcp_trlink = gripper.T_tcp_trlink

    gripper.reset(T_world_pregrasp)
    print(gripper.read())
    T_world_trlink = gripper.body.links['robotiq_arg2f_base_link'].get_pose()
    print((T_world_trlink * T_tcp_trlink.inverse()).as_matrix(), '\n')
    time.sleep(300)

    if gripper.detect_contact():
        result = Label.FAILURE, gripper.max_opening_width
    else:
        gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
        T_world_trlink = gripper.body.links['robotiq_arg2f_base_link'].get_pose()
        print((T_world_trlink * T_tcp_trlink.inverse()).as_matrix(), '\n')
        time.sleep(2)

        if gripper.detect_contact() and not allow_contact:
            result = Label.FAILURE, gripper.max_opening_width
        else:
            gripper.move(0.0)
            gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
            T_world_trlink = gripper.body.links['robotiq_arg2f_base_link'].get_pose()
            print((T_world_trlink * T_tcp_trlink.inverse()).as_matrix(), '\n')
            time.sleep(2)

            if check_success(world, gripper):
                result = Label.SUCCESS, gripper.read()
                if remove:
                    contacts = world.get_contacts(gripper.body)
                    world.remove_body(contacts[0].bodyB)
            else:
                result = Label.FAILURE, gripper.max_opening_width

    print(result)
    print("retreat:", '\n', T_world_retreat.as_matrix())
    # gripper.close()
    print(gripper.read())
    # T_world_origin = T_world_retreat * Transform(Rotation.identity(), [-0.079, 0.15, 0.17])
    # gripper.move_tcp_xyz(T_world_origin)
    # time.sleep(30)
    gripper.move(0.25)
    print(gripper.read())
    print(gripper.link1.get_pose().translation)
    print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    gripper.move(0.5)
    print(gripper.read())
    print(gripper.link1.get_pose().translation)
    print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    gripper.move(0.75)
    print(gripper.read())
    print(gripper.link1.get_pose().translation)
    print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    gripper.move(1.)
    print(gripper.read())
    print(gripper.link1.get_pose().translation)
    print(gripper.link2.get_pose().translation)
    # print(gripper.joint1.get_position(), '\n')
    time.sleep(3)
    # gripper.move(0.)
    print(result)
    print("retreat:", '\n', T_world_retreat.as_matrix())
    # gripper.close()
    print(gripper.read())
    print(gripper.link1.get_pose().translation)
    print(gripper.link2.get_pose().translation)
    gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False)
    gripper.move(0.)
    # print(gripper.joint1.get_position())
    # T_world_origin = T_world_retreat * Transform(Rotation.identity(), [-0.079, 0.15, 0.17])
    # gripper.move_tcp_xyz(T_world_origin)
    time.sleep(30)

"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Hover as close to target position as possible."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.ang_vel = 20.0
        self.lin_acel = 400.0
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2, 0.0,
                      -1.0, -1.0, -1.0, -1.0,
                      -self.ang_vel, -self.ang_vel, -self.ang_vel,
                      -self.lin_acel, -self.lin_acel, -self.lin_acel], dtype=np.float32),
            np.array([cube_size / 2, cube_size / 2, cube_size,
                      1.0, 1.0, 1.0, 1.0,
                      self.ang_vel, self.ang_vel, self.ang_vel,
                      self.lin_acel, self.lin_acel, self.lin_acel], dtype=np.float32)
            , dtype=np.float32)
        #print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_lift = 45.0
        max_force = 5.0
        max_torque = 5.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque], dtype=np.float32),
            np.array([max_force, max_force, max_lift, max_torque, max_torque, max_torque], dtype=np.float32)
            , dtype=np.float32)
        #print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs

        # Target hover position
        self.pos_target_x = 0.0
        self.pos_target_y = 0.0
        self.pos_target_z = 10.0

        self.orientation_target_x = 0.0
        self.orientation_target_y = 0.0
        self.orientation_target_z = 0.0
        self.orientation_target_w = 0.0

        self.termination_boundary = 30.0

        self.starting_twist = Twist(linear=Vector3(0.0, 0.0, 0.0),
                                    angular=Vector3(0.0, 0.0, 0.0))

    def reset(self):
        # Nothing to reset; just return initial condition
        # Start at a random pose, near target hover position
        starting_pose = Pose(position=Point(np.random.normal(self.pos_target_x, 0.5),
                                            np.random.normal(self.pos_target_y, 0.5),
                                            np.random.normal(self.pos_target_z, 0.5)),
                             orientation=Quaternion(0.0, 0.0, 0.0, 0.0))

        # self.termination_boundary = 10.0*(self.squared_error_pos(starting_pose)+0.2)
        # self.termination_boundary = 20.0

        return starting_pose, self.starting_twist

    def pos_loss(self, pose):
        # square of the euclidean distance from current position to target position
        # Just dropping the square root since it really doesn't do much for us.
        loss = (self.pos_target_x - pose.position.x) ** 2
        loss += (self.pos_target_y - pose.position.y) ** 2
        loss += (self.pos_target_z - pose.position.z) ** 2
        return loss

    def orientation_loss(self, pose):
        # square of the euclidean distance from current position to target position
        # Just dropping the square root since it really doesn't do much for us.
        loss = (self.orientation_target_x - pose.orientation.x) ** 2
        loss += (self.orientation_target_y - pose.orientation.y) ** 2
        loss += (self.orientation_target_z - pose.orientation.z) ** 2
        loss += (self.orientation_target_w - pose.orientation.w) ** 2
        return loss

    def angular_velocity_loss(self, angular_velocity):
        # square of the euclidean distance from current position to target position
        # Just dropping the square root since it really doesn't do much for us.

        loss = angular_velocity.x ** 2
        loss += angular_velocity.y ** 2
        loss += angular_velocity.z ** 2
        return loss/(3*self.ang_vel**2)

    def linear_acceleration_loss(self, linear_acceleration):
        # square of the euclidean distance from current position to target position
        # Just dropping the square root since it really doesn't do much for us.

        loss = linear_acceleration.x ** 2
        loss += linear_acceleration.y ** 2
        loss += linear_acceleration.z ** 2
        return loss/(3*self.lin_acel**2)

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
                angular_velocity.x, angular_velocity.y, angular_velocity.z,
                linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

        pos_error = self.pos_loss(pose)

        loss = 1.0 * pos_error
        loss += 1.0 * self.orientation_loss(pose)
        loss += 1.0 * self.angular_velocity_loss(angular_velocity)
        loss += 1.0 * self.linear_acceleration_loss(angular_velocity)

        # Compute reward / penalty and check if this episode is complete
        reward = -loss
        done = False

        # If you drift further away from the target you get a penalty
        if pos_error > self.termination_boundary:
            # Bad dog
            reward -= 10.0

            # Start over
            done = True

            # print(self.max_values)

        # Bonus for being in the hover area!
        elif timestamp > self.max_duration:
            # Good dog
            reward += 10.0  # bonus reward

            # start over
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done

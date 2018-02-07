"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Takeoff(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.ang_vel = 1.5*20.0
        self.lin_acel = 1.5*400.0
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
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_lift = 45.0
        max_force = 5.0
        max_torque = 5.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque], dtype=np.float32),
            np.array([max_force, max_force, max_lift, max_torque, max_torque, max_torque], dtype=np.float32)
            , dtype=np.float32)
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        self.max_values = np.zeros(13)

    def reset(self):
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def height_loss(self, pose):
        return abs(self.target_z - pose.position.z)

    def trajectory_loss(self, pose):
        loss = pose.position.x ** 2
        loss += pose.position.y ** 2
        return min(100, loss)

    def smooth_loss(self, angular_velocity, linear_acceleration):
        loss = (angular_velocity.x / self.ang_vel) ** 2
        loss += (angular_velocity.y / self.ang_vel) ** 2
        loss += (angular_velocity.z / self.ang_vel) ** 2

        loss += (linear_acceleration.x / self.lin_acel) ** 2
        loss += (linear_acceleration.y / self.lin_acel) ** 2
        loss += (linear_acceleration.z / self.lin_acel) ** 2
        return min(100, loss)

    def base_reward(self, pose, angular_velocity, linear_acceleration):
        loss = self.height_loss(pose)
        loss += 0.01 * self.smooth_loss(angular_velocity, linear_acceleration)
        loss += 0.01 * self.trajectory_loss(pose)

        return -loss

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        state = np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
            angular_velocity.x, angular_velocity.y, angular_velocity.z,
            linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

        self.max_values = np.max([state, np.abs(self.max_values)], axis=0)

        # Compute reward / penalty and check if this episode is complete
        done = False
        reward = self.base_reward(pose, angular_velocity, linear_acceleration)

        if pose.position.z >= self.target_z:  # agent has crossed the target height
            reward += 10.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # if done:
        #     print_string = ""
        #     for val in self.max_values:
        #         print_string += "{:8.2f}".format(val)
        #     print(print_string)

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done

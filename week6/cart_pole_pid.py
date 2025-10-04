# Initial code generated with Perplexity AI - edited by JoniK
#
#  For more info see:
#
#     * https://gymnasium.farama.org/environments/classic_control/cart_pole/
#
import gymnasium as gym
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def __call__(self, measured_value, dt):
        error = self.setpoint - measured_value
        self.integral += error * dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

env = gym.make("CartPole-v1", render_mode="human")
pid = PIDController(Kp=10, Ki=0.1, Kd=0.1, setpoint=0.1)

obs, info = env.reset()
pid.reset()


done = False
tot_steps = 0
while not done:
    # CartPole observation: [cart position, cart velocity, pole angle, pole angular velocity]
    tot_steps += 1
    pole_angle = obs[2]
    pole_angle_dot = obs[3]
    dt = 0.02  # CartPole has a fixed 50Hz physics step
    output = pid(pole_angle, dt)
    action = 0 if output < 0 else 1  # 0 = left, 1 = right
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()

print(f'Total number of steps until terminated: {tot_steps}')

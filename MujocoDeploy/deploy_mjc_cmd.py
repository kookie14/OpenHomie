import sys
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR
from common.remote_controller import KeyListener

def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)
    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    
    q_conj = np.array([w, -x, -y, -z])
    
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def compute_observation(d, config, action, cmd, height_cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7:7+n_joints].copy()
    dqj = d.qvel[6:6+n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    
    # Handle default angles padding
    if len(config['default_angles']) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(config['default_angles'])] = config['default_angles']
    else:
        padded_defaults = config['default_angles'][:n_joints]
    
    # Scale the values
    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']
    
    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    
    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action
    
    return single_obs, single_obs_dim

def main():
    config_path = sys.argv[1]
    config = load_config(config_path)
    model = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(model)
    n_joints = d.qpos.shape[0] - 7

    policy = torch.jit.load(config['policy_path'])
    key_listener = KeyListener()
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    height_cmd = config['height_cmd']

    obs_history = collections.deque(maxlen=config['obs_history_len'])
    obs = np.zeros(config['num_obs'], dtype=np.float32)

    # Init history with 
    counter = 0
    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))

    with mujoco.viewer.launch_passive(model, d) as viewer:
        while viewer.is_running():
            if key_listener.is_pressed('q'):
                break
            if key_listener.is_pressed('w'):
                height_cmd += 0.01
                print("[MODE] New command: ", height_cmd)
            if key_listener.is_pressed('s'):
                height_cmd -= 0.01
                print("[MODE] New command: ", height_cmd)
            if key_listener.is_pressed('a'):
                cmd[0] += 0.01
                print("[MODE] New command: ", cmd)
            if key_listener.is_pressed('d'):
                cmd[0] -= 0.01
                print("[MODE] New command: ", cmd)

            if key_listener.is_pressed('z'):
                cmd[1] += 0.01
                print("[MODE] New command: ", cmd)
            if key_listener.is_pressed('x'):
                cmd[1] -= 0.01
                print("[MODE] New command: ", cmd)

            if key_listener.is_pressed('r'):
                cmd[2] += 0.01
                print("[MODE] New command: ", cmd)
            if key_listener.is_pressed('f'):
                cmd[2] -= 0.01
                print("[MODE] New command: ", cmd)
             
            if key_listener.is_pressed('c'):
                # reset the simulation
                cmd = cmd*0.0
   
            # Compute observation and action
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+config['num_actions']],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+config['num_actions']],
                config['kds']
            )
            d.ctrl[:config['num_actions']] = leg_tau

            if n_joints > config['num_actions']:
                arm_kp = 100.0
                arm_kd = 0.5
                n_arms = n_joints - config['num_actions']
                # arm_target_positions = np.zeros(n_joints - config['num_actions'], dtype=np.float32)
                t = d.time  # current simulation time
                arm_freq = 0.5  # 1 Hz oscillation
                arm_amp = 0.5   # radians

                arm_target_positions = arm_amp * np.sin(2 * np.pi * arm_freq * t + np.arange(n_arms))
                arm_tau = pd_control(
                    arm_target_positions,
                    d.qpos[7+config['num_actions']:7+n_joints],
                    np.ones(n_joints-config['num_actions']) * arm_kp,
                    np.zeros(n_joints-config['num_actions']),
                    d.qvel[6+config['num_actions']:6+n_joints],
                    np.ones(n_joints-config['num_actions']) * arm_kd
                )
                
                if d.ctrl.shape[0] > config['num_actions']:
                    d.ctrl[config['num_actions']:] = arm_tau
            mujoco.mj_step(model, d)

            counter += 1
            if counter % config['control_decimation'] == 0:
                single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
                obs_history.append(single_obs)
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs
                
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # Transform action to target_dof_pos
                target_dof_pos = action * config['action_scale'] + config['default_angles']

            viewer.sync()

if __name__ == "__main__":
    main()

import numpy as np
import torch
from model import Network
from gym_simpletetris.envs import TetrisEnv

def test_model(model_file):
  env = TetrisEnv(
        height=20,                       # Height of Tetris grid
        width=10,                        # Width of Tetris grid
        obs_type='ram',                  # ram | grayscale | rgb
        extend_dims=False,               # Extend ram or grayscale dimensions
    #    render_mode='human',         # Unused parameter
        reward_step=True,               # See reward table
        penalise_height=True,           # See reward table
        penalise_height_increase=True,  # See reward table
        advanced_clears=False,           # See reward table
        high_scoring=True,              # See reward table
        penalise_holes=True,            # See reward table
        penalise_holes_increase=False,   # See reward table
        lock_delay=0,                    # Lock delay as number of steps
        step_reset=False                 # Reset lock delay on step downwards
    )
  state = env.reset()
  env.render()

  checkpoint = torch.load(model_file)
  policy_network = Network(env)
  policy_network.load_state_dict(checkpoint['policy'])
  policy_network.eval()

  results = []

  while True:
    try:
      while True:
        observation = torch.tensor(state).float().detach()
        observation.unsqueeze(0)
        with torch.no_grad():       # so we don't compute gradients - save memory and computation
          q_values = policy_network(observation)

        action = torch.argmax(q_values).item()
        state, _, done, info = env.step(action)
        env.render()
        if done:
          break
      state = env.reset()
    except KeyboardInterrupt:
      print("Finished!")
      break

  # print(f'Success rate: {results.count(True)}/{len(results)} ({100*results.count(True)/len(results)}%)')
  try:
    env.close()
  except:
    pass
  

if __name__ == "__main__":
<<<<<<< HEAD
  model_file = "models/240512-201003/policy_network21.pkl"
=======
  model_file = "models\\240512-174516\\policy_network_final_s0.pkl"
>>>>>>> 6225526b47feebfae94df5052c73837a389c2ff4
  test_model(model_file)
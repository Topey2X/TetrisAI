import numpy as np
import torch
from model import Network
from gym_simpletetris.envs import TetrisEnv

def test_model(model_file):
  env = TetrisEnv(renders=True, isDiscrete=True)

  state = np.array(env.getExtendedObservation(), dtype=np.float32)

  checkpoint = torch.load(model_file)
  policy_network = Network(env)
  policy_network.load_state_dict(checkpoint['policy'])
  policy_network.eval()

  results = []

  while True:
    try:
      for _ in range(200):
        observation = torch.tensor(state).float().detach()
        observation.unsqueeze(0)
        with torch.no_grad():       # so we don't compute gradients - save memory and computation
          q_values = policy_network(observation)

        action = torch.argmax(q_values).item()
        state, _, done, info = env.step(action)
        if info['reached_goal'] or info['hit_obstacle']:
          results.append(info['reached_goal'])
          break
      state = env.reset()
    except:
      break

  print(f'Success rate: {results.count(True)}/{len(results)} ({100*results.count(True)/len(results)}%)')
  try:
    env.close()
  except:
    pass
  

if __name__ == "__main__":
  model_file = "models/240415-150753/policy_network_final.pkl"
  test_model(model_file)
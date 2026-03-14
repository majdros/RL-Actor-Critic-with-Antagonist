"""
Sammelt Trainingsdaten aus dem Environment für Actor–Critic.

Ein Rollout entspricht einer vollständigen Episode oder einem
festen Horizont T an Environment-Schritten.

Die Funktion collect_rollout führt eine Episode (oder max. horizon Steps) aus und speichert
alle Größen, die später für das Policy-Update benötigt werden.
"""

import torch
import numpy as np
from finger_env import EnvConfig

device = EnvConfig().device

def collect_rollout(env, actor, critic, horizon, device=device):
    """
    Führt einen Rollout (Episode oder max. horizon Steps) im Environment aus.

    Paramter
    --------
    env: gymnasium environment 
        finger_env.py
    actor: Actor 
        Gauß policy_Netzwertḱ
    critic: State_Value_Funktion Netzwerk
    horizon: int
        Maximale Anzahl von Steps im Rollout (Rollout-Horizon T)
    device: str
        "cpu" oder "cuda"
    
    Returns
    -------

    rollout : dict
        Enthält alle gesammelten Trainingsdaten
    """

    obs_list    = []
    action_list = []
    reward_list = []
    logprob_list= []
    value_list  = []
    entropy_list= []
    done_list   = []

    last_info = {}

    # Environment zurücksetzen
    obs, _ = env.reset()

    for step in range(horizon):
        #observation -> Tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # Actor: Aktion sampeln π(a|s)
        action_tensor, log_prob, entropy = actor.choose_action(obs_tensor)

        # Critic: Zustand bewerten V(s) schätzen
        value = critic(obs_tensor)

        # Aktion im Environment ausführen
        # Actor liefert durch tanh Werte in [-1, 1].
        # Für das Environment muss auf [-max_delta, +max_delta] skaliert werden.
        action_np = (env.cfg.max_delta * action_tensor).detach().cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        last_info = info
        done = terminated or truncated

        # Daten speichern
        obs_list.append(obs)
        action_list.append(action_np)
        reward_list.append(reward)
        
        logprob_list.append(log_prob)
        value_list.append(value)
        entropy_list.append(entropy)

        done_list.append(done)

        # nächster Zustand
        obs = next_obs

        if done:
            break

    # Listen in Tensoren umwandeln
    rollout = dict(
        obs=torch.tensor(np.array(obs_list), dtype=torch.float32, device=device),
        actions=torch.tensor(np.array(action_list), dtype=torch.float32, device=device),
        rewards=torch.tensor(reward_list, dtype=torch.float32, device=device),
        log_probs=torch.stack(logprob_list),
        values=torch.stack(value_list),
        entropies=torch.stack(entropy_list),
        dones=torch.tensor(done_list, dtype=torch.float32, device=device),
        last_info = last_info
    )

    return rollout
# Adapted from CleanRL ppo.py (https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy)
# Changes for "Learning Collaborative and Robust Scheduling Policies for Microservice Backends":
#   * env: TrainTicketSched-v0 (simulator/), MultiDiscrete joint action (Eq.2) -> factorised Categorical
#   * networks: separate policy / value MLPs with hidden (128, 64)  [Table 1]
#   * Adam lr=1e-3, gamma=0.99, minibatch=64, 200 episodes x 100 steps  [Table 1]
#   * robust objective (Eq.6): AdversarialDisturbance wrapper with strength --perturb-eps
#   * constraints (Eq.7): Lagrangian relaxation, r' = r - lambda^T [g]_+, dual ascent on lambda
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

import simulator  # noqa: F401  registers TrainTicketSched-v0
from simulator import AdversarialDisturbance
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "TrainTicketSched-v0"
    """the id of the environment"""
    total_timesteps: int = 0
    """total timesteps (0 -> derived from episodes * max_steps * num_envs)"""
    episodes: int = 200
    """training episodes per env [Table 1]"""
    max_steps: int = 100
    """maximum decision steps per episode [Table 1]"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    minibatch_size_cfg: int = 64
    """minibatch size [Table 1 'Batch Size']"""
    perturb_eps: float = 0.3
    """abnormal perturbation strength coefficient (0 disables the robust adversary)"""
    adv_k: int = 4
    """candidate disturbances per step for the inner min of Eq.(6)"""
    lagrangian: bool = True
    """enable constraint handling of Eq.(7)"""
    lambda_lr: float = 0.05
    """dual ascent step size"""
    save_path: str = ""
    """where to save the trained agent (state_dict)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, max_steps=100, perturb_eps=0.0, adv_k=4):
    def thunk():
        env = gym.make(env_id, max_steps=max_steps)
        if perturb_eps > 0:
            env = AdversarialDisturbance(env, eps=perturb_eps, k=adv_k)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def merged_field(infos, key, n_envs, shape):
    """Collect a per-env info field under gymnasium>=1.0 SAME_STEP autoreset (final steps live in final_info)."""
    out = np.zeros((n_envs,) + shape)
    for src in (infos, infos.get("final_info", {})):
        if key in src:
            mask = src[f"_{key}"]
            out[mask] = np.asarray(src[key], dtype=np.float64).reshape((n_envs,) + shape)[mask]
    return out


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Single policy network + single value network, hidden (128, 64) [Table 1]."""

    def __init__(self, obs_dim, nvec):
        super().__init__()
        self.nvec = [int(n) for n in nvec]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, sum(self.nvec)), std=0.01),
        )
        # bias the scaling / quota heads towards "keep" (index 1) so the initial policy is near no-op
        with torch.no_grad():
            b = self.actor[-1].bias.view(len(self.nvec), -1) if len(set(self.nvec)) == 1 else None
            if b is not None:
                b[:, 1] += 1.5

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        dists = [Categorical(logits=l) for l in torch.split(logits, self.nvec, dim=1)]
        if action is None:
            action = torch.stack([d.probs.argmax(1) if deterministic else d.sample() for d in dists], dim=1)
        logprob = torch.stack([d.log_prob(a) for d, a in zip(dists, action.T)], dim=1).sum(1)
        entropy = torch.stack([d.entropy() for d in dists], dim=1).sum(1)
        return action, logprob, entropy, self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_steps = args.max_steps
    if args.total_timesteps <= 0:
        args.total_timesteps = args.episodes * args.max_steps * args.num_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = min(args.minibatch_size_cfg, args.batch_size)
    args.num_iterations = max(args.total_timesteps // args.batch_size, 1)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"outputs/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.max_steps, args.perturb_eps, args.adv_k)
         for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.MultiDiscrete), "joint MultiDiscrete action expected"

    agent = Agent(int(np.prod(envs.single_observation_space.shape)), envs.single_action_space.nvec).to(device)
    n_cons = 3
    lam = np.zeros(n_cons)                         # Lagrange multipliers for Eq.(7)
    cons_buf = np.zeros((args.num_steps, args.num_envs, n_cons))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            g = merged_field(infos, "constraints", args.num_envs, (n_cons,))
            cons_buf[step] = g
            if args.lagrangian:
                reward = reward - (np.maximum(g, 0.0) * lam).sum(1)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # gymnasium>=1.0 vector API: episode stats are in infos["episode"] masked by infos["_episode"]
            fin = infos.get("final_info", {})
            if "episode" in fin:
                for i in np.where(fin["_episode"])[0]:
                    print(f"global_step={global_step}, episodic_return={float(fin['episode']['r'][i]):.2f}, "
                          f"lambda={np.round(lam, 3)}")
                    writer.add_scalar("charts/episodic_return", fin["episode"]["r"][i], global_step)
                    writer.add_scalar("charts/episodic_length", fin["episode"]["l"][i], global_step)

        # dual ascent on Lagrange multipliers (Eq.7): lambda <- [lambda + eta * mean(g)]_+
        if args.lagrangian:
            lam = np.maximum(lam + args.lambda_lr * cons_buf.reshape(-1, n_cons).mean(0), 0.0)
            for k in range(n_cons):
                writer.add_scalar(f"constraints/lambda_{k}", lam[k], global_step)
                writer.add_scalar(f"constraints/g_{k}", cons_buf[..., k].mean(), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({"state_dict": agent.state_dict(), "obs_dim": int(np.prod(envs.single_observation_space.shape)),
                    "nvec": [int(n) for n in envs.single_action_space.nvec], "args": vars(args)}, args.save_path)
        print(f"saved agent -> {args.save_path}")
    envs.close()
    writer.close()

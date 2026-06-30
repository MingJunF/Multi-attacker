"""Stage-aware continuous Q critic (off-policy Stage-MADDPG).

This is the off-policy analogue of the Stage-Aware MAPPO critic. It keeps a
SINGLE shared continuous Q network but gives it a genuine two-stage structure
driven by the centralized state and a coupled, stage-decomposed Bellman target.

Two ordered decision stages per env step (same victim, same shared Q):
    o-stage (observation attacker, leader):  x^o = [s, 0, onehot_o]
    a-stage (action attacker,    follower):  x^a = [s, victim.act(s+delta_o), onehot_a]

The centralized Q in MADDPG conditions on the JOINT action [delta_o, delta_a].
To keep the o-stage value leakage-free (it must not peek at the downstream
action attack), the obs-stage Q input masks delta_a to zero:
    Q^a := Q(x^a, [delta_o, delta_a])     (act-stage value, full joint action)
    Q^o := Q(x^o, [delta_o,        0])    (obs-stage value, delta_a masked out)

Coupled one-step (TD0, n_step=1) targets, mirroring the Stage-MAPPO returns
(returns[a] = r + gamma V^o', returns[o] = (1-lam) V^a + lam (r + gamma V^o')):
    q_o_next = Q_target(x^o_{t+1}, [delta_o', 0])
    y_a      = r + gamma * cont * q_o_next
    q_a_targ = Q_target(x^a_t, [delta_o, delta_a])
    y_o      = (1 - stage_lambda) * q_a_targ + stage_lambda * y_a
with stage_lambda in [0, 1]: 1 -> full downstream coupling (y_o = y_a),
0 -> pure intra-step obs value (y_o = Q^a, leakage-free, most myopic).
"""
import torch

from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.utils.envs_tools import check


class StageQCritic(ContinuousQCritic):
    """Single shared continuous Q critic with a coupled two-stage target.

    Assumes exactly two ordered agents (leader = obs attacker, follower = act
    attacker) and an FP centralized state whose per-agent rows encode the stage
    (x^o for the leader, x^a for the follower). The FP off-policy buffer samples
    ``share_obs`` agent-major, i.e. rows ``[0:B]`` belong to the leader (x^o) and
    rows ``[B:2B]`` to the follower (x^a).
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        super().__init__(
            args, share_obs_space, act_space, num_agents, state_type, device
        )
        assert num_agents == 2, "Stage-MADDPG assumes two ordered agents."
        assert state_type == "FP", (
            "Stage-MADDPG requires state_type: FP (per-agent stage state)."
        )
        self.leader_id = 0
        self.follower_id = 1
        stage_lambda = args.get("stage_lambda", None)
        if stage_lambda is None:
            stage_lambda = 0.95
        self.stage_lambda = float(stage_lambda)

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Train the shared critic with the coupled two-stage TD target.

        Args:
            share_obs: (np.ndarray) FP, (n_agents * batch, dim), agent-major.
            actions: (np.ndarray) (n_agents, batch, act_dim).
            reward/done/term/gamma: (np.ndarray) FP, (n_agents * batch, 1).
            next_share_obs: (np.ndarray) FP, (n_agents * batch, dim).
            next_actions: (list[Tensor]) per-agent target actions, each
                (batch, act_dim); only the leader's is used (next obs stage).
        """
        share_obs = check(share_obs).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)  # (n_agents, batch, act_dim)

        batch = actions.shape[1]
        delta_o = actions[self.leader_id]  # (batch, act_dim)
        delta_a = actions[self.follower_id]  # (batch, act_dim)
        joint = torch.cat([delta_o, delta_a], dim=-1)  # (batch, 2*act_dim)
        joint_o = torch.cat(
            [delta_o, torch.zeros_like(delta_a)], dim=-1
        )  # obs stage: mask delta_a -> leakage-free

        # FP rows are agent-major: leader (x^o) first, follower (x^a) second.
        x_o = share_obs[:batch]
        x_a = share_obs[batch : 2 * batch]
        next_x_o = next_share_obs[:batch]
        # reward / done / term / gamma are identical across the two cooperative
        # agents, so the leader rows suffice.
        r = reward[:batch]
        d = done[:batch]
        tm = term[:batch]
        g = gamma[:batch]

        next_delta_o = check(next_actions[self.leader_id]).to(**self.tpdv)
        next_joint_o = torch.cat(
            [next_delta_o, torch.zeros_like(next_delta_o)], dim=-1
        )

        with torch.no_grad():
            cont = (1 - tm) if self.use_proper_time_limits else (1 - d)
            # y_a bootstraps the NEXT obs-stage value (across-step, reward + discount).
            q_o_next = self.target_critic(next_x_o, next_joint_o)
            y_a = r + g * q_o_next * cont
            # y_o couples the within-step act-stage value with the bootstrapped
            # return via stage_lambda (intra-step, no extra reward/discount).
            q_a_targ = self.target_critic(x_a, joint)
            y_o = (1.0 - self.stage_lambda) * q_a_targ + self.stage_lambda * y_a

        q_a_pred = self.critic(x_a, joint)
        q_o_pred = self.critic(x_o, joint_o)
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(q_a_pred, y_a)
        ) + torch.mean(torch.nn.functional.mse_loss(q_o_pred, y_o))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

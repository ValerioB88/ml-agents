from typing import Dict, cast
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import TrainerSettings, PPOSettings
from mlagents.trainers.mltorch.networks import ValueNetwork
from mlagents.trainers.mltorch.agent_action import AgentAction
from mlagents.trainers.mltorch.action_log_probs import ActionLogProbs
from mlagents.trainers.mltorch.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


class TorchPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        ## Valerio Addition
        self.connection_cost = trainer_settings.connection_cost
        if self.connection_cost == 'linear':
            self.connection_cost_mat = torch.tensor([[abs(i - idx) + 1 for i in range(trainer_settings.network_settings.hidden_units)] for idx in range( trainer_settings.network_settings.hidden_units)])
            print("Connection Cost Added")
        else:
            print("No Connection Cost")
        self.connection_cost_lambda = 0.0000005
        print(f"Connection cost lambda: {self.connection_cost_lambda}")

        if policy.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())

        params = list(self.policy.actor.parameters()) + list(self._critic.parameters())
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)

        ############### DEBUG
        import matplotlib.pyplot as plt
        plt.imshow(current_obs[0][510])

        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        log_probs, entropy = self.policy.evaluate_actions(
            current_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )
        connection_loss = 0
        # The cost depends on the size of the hidden layer, but does not depend on the depth of the network (it's averaged across num of hidden layers)
        count = 0
        if self.connection_cost is not None:
            for n, p in self.policy.actor.network_body._body_endoder.seq_layers.named_parameters():
                if p.requires_grad and ("bias" not in n):
                    # print(p)
                    # pp = p
                    count += 1
                    connection_loss += torch.sum(torch.abs(p) * self.connection_cost_mat)

            connection_loss = self.connection_cost_lambda * connection_loss / count
        loss = (
            policy_loss
            + 0.5 * value_loss
            + connection_loss
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        print(f"[LOSS INFO] PolicyL: {policy_loss}, ValueL: {value_loss}, ConnectionL: {connection_loss}")

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Tot Loss": loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,

        }
        if self.connection_cost is not None:
            update_stats.update({"Losses/Connection Cost Loss": connection_loss.item()})

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

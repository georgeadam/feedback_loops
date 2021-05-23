import logging
import numpy as np
import torch

from src.rl.envs.basic_icu import action_mapping
from src.utils.metrics import compute_rl_rates

logger = logging.getLogger(__name__)


def nfqi(env, agent, decay_fn, optimizer, gamma, epochs, writer, mode):
    env.set_mode(mode)

    for epoch in range(1, epochs + 1):
        state, label = env.reset()
        samples = 0

        actions = []
        labels = []
        occupancies = []

        ys = torch.zeros(env.x.shape[0])
        preds = torch.zeros(env.x.shape[0])

        while True:
            action, pred = agent.act(state)
            old_label = label
            old_state = state
            state, reward, done, label = env.step(action)

            # if epoch % 10 == 0:
            #     logger.info("Epoch: {} | Action: {:<10} | Label: {:<10} | Reward: {:<5} | ICU %: {}".format(epoch,
            #                                                                                                 action_mapping[action],
            #                                                                                                 action_mapping[old_label],
            #                                                                                                 reward, old_state[0]))

            if done:
                y = reward
            else:
                q = agent.q(state)
                max_q = torch.max(q, 1)[0]
                y = reward + gamma * max_q

            if done:
                break

            ys[samples] = y
            preds[samples] = (pred[:, action])

            actions.append(action)
            labels.append(old_label)
            occupancies.append(old_state[0])

            samples += 1

        loss = torch.mean((ys - preds) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        actions = np.array(actions)
        labels = np.array(labels)
        occupancies = np.array(occupancies)

        non_full_idx = occupancies != 1.0
        all_rates = compute_rl_rates(labels, actions)
        relevant_rates = compute_rl_rates(labels[non_full_idx], actions[non_full_idx])

        if epoch % 100 == 0:
            logger.info("****************** Loss: {} ******************".format(loss))
            logger.info("All rates: {}".format(all_rates))
            logger.info("Relevant rates: {}".format(relevant_rates))

        eps = decay_fn.step(epoch)
        agent.eps = eps
        # writer.add_scalar("train_loss", loss, epoch)

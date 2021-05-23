import hydra
import logging
import numpy as np
import os
import pandas as pd
import torch
from omegaconf import DictConfig
from settings import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter

from src.rl.agents.epsilon_greedy import EpsilonGreedyAgent
from src.rl.algorithms import get_algorithm
from src.rl.envs.basic_icu import BasicICUEnvironment
from src.utils.data import get_data_fn
from src.utils.decay import get_epsilon_decay_fn
from src.utils.optimizer import create_optimizer
from src.utils.metrics import compute_rl_rates
from src.utils.model import get_model_fn
from src.utils.preprocess import get_scaler
from src.utils.rand import set_seed
from src.utils.save import CSV_FILE


logger = logging.getLogger(__name__)


def eval_agent_hardcoded(env, agent, all_stats, seed, update_num):
    agent.eps = 0.0

    labels = env.y_test
    x = env.x_test

    if env.include_icu:
        x = np.concatenate([np.zeros(len(x)).reshape(-1, 1), x], axis=1)

    actions = torch.max(agent.q(x), 1)[1].detach().cpu().numpy()

    all_rates = compute_rl_rates(labels, actions)
    logger.info("All rates: {}".format(all_rates))

    for key in all_rates.keys():
        all_stats = all_stats.append({"rate_group": "all", "rate_type": key, "rate": all_rates[key], "seed": seed, "update_num": update_num}, ignore_index=True)

    return all_stats


def eval_agent(env, agent, all_stats, seed, update_num, runs):
    env.set_mode("test")
    agent.eps = 0.0

    for run in range(1, runs + 1):
        state, label = env.reset()
        samples = 0

        actions = []
        labels = []
        occupancies = []

        while True:
            action, pred = agent.act(state)
            old_label = label
            old_state = state
            state, reward, done, label = env.step(action)

            samples += 1

            actions.append(action)
            labels.append(old_label)
            occupancies.append(old_state[0])

            if done:
                break

        actions = np.array(actions)
        labels = np.array(labels)
        occupancies = np.array(occupancies)

        non_full_idx = occupancies != 1.0
        all_rates = compute_rl_rates(labels, actions)
        relevant_rates = compute_rl_rates(labels[non_full_idx], actions[non_full_idx])
        logger.info("All rates: {}".format(all_rates))
        logger.info("Relevant rates: {}".format(relevant_rates))

        for key in all_rates.keys():
            all_stats = all_stats.append({"run": run, "rate_group": "all", "rate_type": key, "rate": all_rates[key], "seed": seed, "update_num": update_num}, ignore_index=True)
            all_stats = all_stats.append({"run": run, "rate_group": "relevant", "rate_type": key, "rate": relevant_rates[key], "seed": seed, "update_num": update_num}, ignore_index=True)

    return all_stats


os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
@hydra.main(config_path=config_path, config_name="rl_feedback")
def main(args: DictConfig):
    print(args.pretty())
    print("Saving to: {}".format(os.getcwd()))

    inner_data_fn = get_data_fn(args)
    data_fn = lambda: inner_data_fn(args.data.n_train, args.data.n_val, args.data.n_update, args.data.n_test,
                                    args.data.num_features)
    algorithm = get_algorithm(args.algorithm.type)
    all_stats = pd.DataFrame({"run": {}, "rate_group": {}, "rate_type": {}, "rate": {}, "seed": {}, "update_num": {}})

    for seed in range(args.misc.seeds):
        set_seed(seed)
        data, cols = data_fn()
        scaler = get_scaler(True, cols)

        env = BasicICUEnvironment(data, args.data.num_updates, scaler, args.env.icu_limit, args.env.include_icu,
                                  args.env.r_fn, args.env.r_fp,
                                  args.env.r_tn, args.env.r_tp, args.env.reward_fn,
                                  args.env.alpha, args.env.beta)
        decay_fn = get_epsilon_decay_fn(args.agent.decay, args.optim.epochs)
        eps = decay_fn.step(0)
        gamma = args.algorithm.gamma

        q_net = get_model_fn(args)
        q_net = q_net(env.dimension)

        agent = EpsilonGreedyAgent(q_net, eps)
        optimizer = create_optimizer(q_net.parameters(), args.optim.optimizer, args.optim.lr, args.optim.momentum,
                                     args.optim.nesterov, args.optim.weight_decay)
        # writer = SummaryWriter("tensorboard_logs/{}".format(seed))
        writer = None

        algorithm(env, agent, decay_fn, optimizer, gamma, args.optim.epochs, writer, "train")

        set_seed(0)
        all_stats = eval_agent_hardcoded(env, agent, all_stats, seed, 0)

        for update_num in range(args.data.num_updates):
            logger.info("Now on seed: {} | Update Num: {}".format(seed, update_num))
            x_update, y_update = env.get_update_data(update_num)
            preds = torch.max(q_net(scaler.transform(x_update)), 1)[1].detach().cpu().numpy()

            if args.update_params.feedback:
                fp_idx = (y_update == 0) & (preds == 1)
                y_update[fp_idx] = 1

            env.store_update_data(x_update[:, 1:], y_update)
            algorithm(env, agent, decay_fn, optimizer, gamma, args.optim.epochs, writer, "update")

            set_seed(0)
            all_stats = eval_agent_hardcoded(env, agent, all_stats, seed, update_num + 1)

    all_stats.to_csv(CSV_FILE, index=False, header=True)


if __name__ == "__main__":
    main()
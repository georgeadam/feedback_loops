def train_lre(model, x_train, y_train, x_val_reg, y_val_reg, x_val_lre, y_val_lre, optimizer, model_args,
              optim_args, writer, writer_prefix, device="cpu"):
    meta_losses_clean = []
    net_losses = []
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = copy.deepcopy(model.state_dict())
    no_val_improvement = 0
    no_train_improvement = 0

    smoothing_alpha = 0.9
    meta_l = 0
    net_l = 0
    done = False
    epoch = 0

    while not done:
        model.train()
        # Line 2 get batch of data
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_model = NN_LRE(x_train.shape[1], 2, model_args.hidden_layers, model_args.activation, device)
        meta_model.load_state_dict(model.state_dict())
        meta_model = meta_model.to(x_train.device)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_model(x_train)
        cost = F.cross_entropy(y_f_hat, y_train, reduce=False)
        eps = torch.zeros(cost.size()).to(x_train.device).requires_grad_(True)

        l_f_meta = torch.sum(cost * eps)

        meta_model.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(optim_args.lr, source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_model(x_val_lre)

        l_g_meta = F.cross_entropy(y_g_hat, y_val_lre)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = model(x_train)
        cost = F.cross_entropy(y_f_hat, y_train, reduce=False)
        l_f = torch.sum(cost * w)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        with torch.no_grad():
            unweighted_loss = F.cross_entropy(y_f_hat, y_train)

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (epoch + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (epoch + 1)))

        with torch.no_grad():
            val_loss = compute_loss(model, x_val_reg, y_val_reg, F.cross_entropy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = unweighted_loss
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())
            no_val_improvement = 0
        else:
            no_val_improvement += 1

        if unweighted_loss < best_train_loss:
            no_train_improvement = 0
        elif no_val_improvement > 0:
            no_train_improvement += 1

        if no_train_improvement > optim_args.early_stopping_iter:
            done = True
            logger.info("No improvement in train loss for {} epochs at epoch: {}. Stopping.".format(optim_args.early_stopping_iter,
                                                                                                    epoch))

        if no_val_improvement > optim_args.early_stopping_iter:
            done = True
            logger.info("No improvement in validation loss for {} epochs at epoch: {}. Stopping.".format(optim_args.early_stopping_iter,
                                                                                                         epoch))

        if epoch % 100 == 0:
            logger.info("Epoch: {} | Weighted Train Loss: {} | Unweighted Train Loss: {} | LRE Val Loss: {} | Reg Val Loss: {}".format(epoch,
                                                                                                                                       l_f.item(),
                                                                                                                                       unweighted_loss.item(),
                                                                                                                                       l_g_meta.item(),
                                                                                                                                       val_loss.item()))

        if epoch > optim_args.epochs:
            break

        log_lre_losses(writer, writer_prefix, unweighted_loss.item(), val_loss.item(), l_g_meta.item(), epoch)

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Train Loss: {} | Best LRE Val Loss: {} | Best Reg Val Loss Achieved: {} at epoch: {}".format(best_train_loss.item(),
                                                                                                                   best_val_loss.item(),
                                                                                                                   l_g_meta.item(),
                                                                                                                   best_epoch))

    if not done:
        logger.info("Stopped after: {} epochs, but could have kept improving loss.".format(optim_args.epochs))

    return meta_losses_clean, net_losses, w


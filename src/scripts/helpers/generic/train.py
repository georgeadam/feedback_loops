# import copy
# import logging
# import torch
# from sklearn.model_selection import train_test_split
#
# from src.utils.data import get_data_fn
# from src.utils.metrics import compute_all_rates
# from src.utils.preprocess import get_scaler
#
# logger = logging.getLogger(__name__)
#
#
# def load_data(data_args, model_args):
#     data_fn = get_data_fn(data_args, model_args)
#     x_train, y_train, x_update, y_update, x_test, y_test, cols = data_fn(data_args.n_train, data_args.n_update,
#                                                                          data_args.n_test,
#                                                                          num_features=data_args.num_features)
#     x_update, x_val, y_update, y_val = train_test_split(x_update, y_update, test_size=0.4)
#     x_val_reg, x_val_lre, y_val_reg, y_val_lre = train_test_split(x_val, y_val, test_size=0.5)
#
#     scaler = get_scaler(True, cols)
#     scaler.fit(x_train)
#
#     x_train_torch = torch.from_numpy(scaler.transform(x_train)).float().to(model_args.device)
#     x_val_reg_torch = torch.from_numpy(scaler.transform(x_val_reg)).float().to(model_args.device)
#     x_val_lre_torch = torch.from_numpy(scaler.transform(x_val_lre)).float().to(model_args.device)
#     x_update_torch = torch.from_numpy(scaler.transform(x_update)).float().to(model_args.device)
#     x_test_torch = torch.from_numpy(scaler.transform(x_test)).float().to(model_args.device)
#
#     y_train_torch = torch.from_numpy(y_train).long().to(model_args.device)
#     y_val_reg_torch = torch.from_numpy(y_val_reg).long().to(model_args.device)
#     y_val_lre_torch = torch.from_numpy(y_val_lre).long().to(model_args.device)
#     y_update_torch = torch.from_numpy(y_update).long().to(model_args.device)
#     y_test_torch = torch.from_numpy(y_test).long().to(model_args.device)
#
#     data = {"x_train": x_train_torch, "y_train": y_train_torch, "x_val_reg": x_val_reg_torch, "y_val_reg": y_val_reg_torch,
#             "x_val_lre": x_val_lre_torch, "y_val_lre": y_val_lre_torch, "x_update": x_update_torch, "y_update": y_update_torch,
#             "x_test": x_test_torch, "y_test": y_test_torch}
#
#     return data
#
#
# def eval_model(model, x, y):
#     model.eval()
#     softmax = torch.nn.Softmax(dim=1)
#     y_out = model(x)
#     y_prob = softmax(y_out)
#     y_pred = torch.max(y_out, 1)[1]
#
#     rates = compute_all_rates(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy(),
#                                         y_prob.detach().cpu().numpy())
#     return rates
#
#
# def compute_loss(model, x, y, criterion):
#     out = model(x)
#
#     return criterion(out, y)
#
#
# def create_corrupted_labels(model, x, y):
#     y_out = model(x)
#     y_pred = torch.max(y_out, 1)[1]
#
#     fps = torch.where((y == 0) & (y_pred == 1))[0]
#
#     y_corrupt = copy.deepcopy(y)
#     y_corrupt[fps] = 1
#     y_corrupt_torch = y_corrupt.long().to(y.device)
#
#     return y_corrupt_torch
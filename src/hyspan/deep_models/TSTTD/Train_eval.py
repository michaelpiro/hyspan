from typing import Dict
import torch
from Data import Data
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
from Model import SpectralGroupAttention
import os
from Tools import checkFile, standard
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn import metrics
import random

from src.hyspan.deep_models.ts_generation import ts_generation


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def spectral_group(x, n, m):
    ### divide the spectrum into n overlapping groups
    pad_size = m // 2
    new_sample = np.pad(x, ((0, 0), (pad_size, pad_size)),
                        mode='symmetric')
    b = x.shape[0]
    group_spectra = np.zeros([b, n, m])
    for i in range(n):
        group_spectra[:, i, :] = np.squeeze(new_sample[:, i:i + m])

    return torch.from_numpy(group_spectra).float()


def cosin_similarity(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = torch.sqrt(torch.sum(x ** 2, dim=1))
    y_norm = torch.sqrt(torch.sum(y ** 2, dim=1))
    x_y_dot = torch.sum(torch.multiply(x, y), dim=1)
    return x_y_dot / (x_norm * y_norm + 1e-8)


def cosin_similarity_numpy(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y = np.sum(np.multiply(x, y), axis=1)
    return x_y / (x_norm * y_norm + 1e-8)


def isia_loss(x, batch_size, margin=1.0, lambd=1):
    '''
    This function is used to calculate the intercategory separation and intracategory aggregation loss
    It includes the triplet loss and cross-entropy loss
    '''
    positive, negative, prior = x[:batch_size], x[batch_size:2 * batch_size], x[2 * batch_size:]
    p_sim = cosin_similarity(positive, prior)
    n_sim1 = cosin_similarity(negative, prior)
    n_sim2 = cosin_similarity(negative, positive)
    max_n_sim = torch.maximum(n_sim1, n_sim2)

    ## triplet loss to maximize the feature distance between anchor and positive samples
    ## while minimizing the feature distance between anchor and negative samples
    triplet_loss = margin + max_n_sim - p_sim
    triplet_loss = torch.relu(triplet_loss)
    triplet_loss = torch.mean(triplet_loss)

    ## binary cross-entropy loss to distinguish pixels of background and target
    p_sim = torch.sigmoid(p_sim)
    n_sim = torch.sigmoid(1 - n_sim1)
    bce_loss = -0.5 * torch.mean(torch.log(p_sim + 1e-8) + torch.log(n_sim + 1e-8))

    isia_loss = triplet_loss + lambd * bce_loss

    return isia_loss


def paintTrend(losslist, epochs=100, stride=10):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title('loss-trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color='r')
    plt.show()


# TODO: VERIFY THAT THE CODE AFTER TRAINING MATCHES THE FOLLOWING OLD ORIGINAL CODE
# def train(modelConfig: Dict):
#     seed_torch(modelConfig['seed'])
#     device = torch.device(modelConfig["device"])
#     dataset = Data(modelConfig["path"])
#     dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True,
#                             pin_memory=True)
#     # model setup
#     net_model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
#                                        d=modelConfig['channel'], depth=modelConfig['depth'],heads=modelConfig['heads'],
#                                        dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'], adjust=modelConfig['adjust']).to(device)
#
#     if modelConfig["training_load_weight"] is not None:
#         net_model.load_state_dict(torch.load(os.path.join(
#             modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
#         print("Model weight load down.")
#     optimizer = torch.optim.AdamW(
#         net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
#     cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
#     warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
#                                              warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
#     path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
#     checkFile(path)
#
#     # start training
#     net_model.train()
#     loss_list = []
#     for e in range(modelConfig["epoch"]):
#         with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
#             for positive, negative in tqdmDataLoader:
#                 # train
#                 combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum], axis=0)
#                 combined_groups = spectral_group(combined_vectors, modelConfig['band'], modelConfig['group_length'])
#                 optimizer.zero_grad()
#                 x_0 = combined_groups.to(device)
#                 features = net_model(x_0)
#                 loss = isia_loss(features, modelConfig['batch_size'])
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(
#                     net_model.parameters(), modelConfig["grad_clip"])
#                 optimizer.step()
#                 tqdmDataLoader.set_postfix(ordered_dict={
#                     "epoch": e,
#                     "loss: ": loss.item(),
#                     "LR": optimizer.state_dict()['param_groups'][0]["lr"]
#                 })
#         warmUpScheduler.step()
#         torch.save(net_model.state_dict(), os.path.join(
#             path, 'ckpt_' + str(e) + "_.pt"))
#         loss_list.append(loss.item())
#     paintTrend(loss_list, epochs=modelConfig['epoch'], stride=5)
#
# def select_best(modelConfig: Dict):
#     seed_torch(modelConfig['seed'])
#     device = torch.device(modelConfig["device"])
#     opt_epoch = 0
#     max_auc = 0
#     path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
#     for e in range(modelConfig['epoch']):
#         with torch.no_grad():
#             mat = sio.loadmat(modelConfig["path"])
#             data = mat['data']
#             map = mat['map']
#             data = standard(data)
#             data = np.float32(data)
#             target_spectrum = ts_generation(data, map, 7)
#             h, w, c = data.shape
#             numpixel = h * w
#             data_matrix = np.reshape(data, [-1, c], order='F')
#             model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
#                                            d=modelConfig['channel'], depth=modelConfig['depth'],
#                                            heads=modelConfig['heads'],
#                                            dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'],
#                                            adjust=modelConfig['adjust']).to(device)
#             ckpt = torch.load(os.path.join(
#                 path, "ckpt_%s_.pt" % e), map_location=device)
#             model.load_state_dict(ckpt)
#             print("model load weight done.%s" % e)
#             model.eval()
#
#             batch_size = modelConfig['batch_size']
#             detection_map = np.zeros([numpixel])
#             target_prior = spectral_group(target_spectrum.T, modelConfig['band'], modelConfig['group_length'])
#             target_prior = target_prior.to(device)
#             target_features = model(target_prior)
#             target_features = target_features.cpu().detach().numpy()
#
#             for i in range(0, numpixel - batch_size, batch_size):
#                 pixels = data_matrix[i:i + batch_size]
#                 pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
#                 pixels = pixels.to(device)
#                 features = model(pixels)
#                 features = features.cpu().detach().numpy()
#                 detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)
#
#             left_num = numpixel % batch_size
#             if left_num != 0:
#                 pixels = data_matrix[-left_num:]
#                 pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
#                 pixels = pixels.to(device)
#                 features = model(pixels)
#                 features = features.cpu().detach().numpy()
#                 detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)
#
#             detection_map = np.reshape(detection_map, [h, w], order='F')
#             detection_map = standard(detection_map)
#             detection_map = np.clip(detection_map, 0, 1)
#             y_l = np.reshape(map, [-1, 1], order='F')
#             y_p = np.reshape(detection_map, [-1, 1], order='F')
#
#             ## calculate the AUC value
#             fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
#             fpr = fpr[1:]
#             tpr = tpr[1:]
#             auc = round(metrics.auc(fpr, tpr), modelConfig['epision'])
#             if auc > max_auc:
#                 max_auc = auc
#                 opt_epoch = e
#     print(max_auc)
#     print(opt_epoch)
#
# def eval(modelConfig: Dict):
#     seed_torch(modelConfig['seed'])
#     device = torch.device(modelConfig["device"])
#     path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
#     with torch.no_grad():
#         mat = sio.loadmat(modelConfig["path"])
#         data = mat['data']
#         map = mat['map']
#         data = standard(data)
#         data = np.float32(data)
#         target_spectrum = ts_generation(data, map, 7)
#         h, w, c = data.shape
#         numpixel = h * w
#         data_matrix = np.reshape(data, [-1, c], order='F')
#         model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
#                                        d=modelConfig['channel'], depth=modelConfig['depth'],heads=modelConfig['heads'],
#                                        dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'], adjust=modelConfig['adjust']).to(device)
#         ckpt = torch.load(os.path.join(
#             path, modelConfig["test_load_weight"]), map_location=device)
#         model.load_state_dict(ckpt)
#         print("model load weight done.")
#         model.eval()
#
#         batch_size = modelConfig['batch_size']
#         detection_map = np.zeros([numpixel])
#         target_prior = spectral_group(target_spectrum.T, modelConfig['band'], modelConfig['group_length'])
#         target_prior = target_prior.to(device)
#         target_features = model(target_prior)
#         target_features = target_features.cpu().detach().numpy()
#
#         for i in range(0, numpixel - batch_size, batch_size):
#             pixels = data_matrix[i:i + batch_size]
#             pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
#             pixels = pixels.to(device)
#             features = model(pixels)
#             features = features.cpu().detach().numpy()
#             detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)
#
#         left_num = numpixel % batch_size
#         if left_num != 0:
#             pixels = data_matrix[-left_num:]
#             pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
#             pixels = pixels.to(device)
#             features = model(pixels)
#             features = features.cpu().detach().numpy()
#             detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)
#
#         detection_map = np.reshape(detection_map, [h, w], order='F')
#         detection_map = standard(detection_map)
#         detection_map = np.clip(detection_map, 0, 1)
#         # plt.imshow(detection_map)
#         # plt.show()
#         y_l = np.reshape(map, [-1, 1], order='F')
#         y_p = np.reshape(detection_map, [-1, 1], order='F')
#
#         ## calculate the AUC value
#         fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
#         fpr = fpr[1:]
#         tpr = tpr[1:]
#         threshold = threshold[1:]
#         auc1 = round(metrics.auc(fpr, tpr), modelConfig['epision'])
#         auc2 = round(metrics.auc(threshold, fpr), modelConfig['epision'])
#         auc3 = round(metrics.auc(threshold, tpr), modelConfig['epision'])
#         auc4 = round(auc1 + auc3 - auc2, modelConfig['epision'])
#         auc5 = round(auc3 / auc2, modelConfig['epision'])
#         print('{:.{precision}f}'.format(auc1, precision=modelConfig['epision']))
#         print('{:.{precision}f}'.format(auc2, precision=modelConfig['epision']))
#         print('{:.{precision}f}'.format(auc3, precision=modelConfig['epision']))
#         print('{:.{precision}f}'.format(auc4, precision=modelConfig['epision']))
#         print('{:.{precision}f}'.format(auc5, precision=modelConfig['epision']))
#
#         plt.imshow(detection_map)
#         plt.show()


# ==========================================
# REFACTORED TRAINING LOGIC
# ==========================================

def setup_training(config: Dict):
    """Initializes device, data, model, and optimizers."""
    seed_torch(config['seed'])
    device = torch.device(config["device"])

    # Note: You will need to update Data.py to accept data_keys
    dataset = Data(
        config["paths"]["dataset_path"],
        data_key=config['data_keys']['data'],
        map_key=config['data_keys']['map']
    )
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4,
                            drop_last=True, pin_memory=True)
    band, m, d, depth, heads, dim_head, mlp_dim, adjust = config['model']['band'], config['model']['group_length'], \
    config['model']['channel'], config['model']['depth'], config['model']['heads'], config['model']['dim_head'], \
    config['model']['mlp_dim'], config['model']['adjust']
    net_model = SpectralGroupAttention(band=band, m=m, d=d, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                       adjust=adjust).to(device)

    if config["paths"]["training_load_weight"]:
        weight_path = os.path.join(config["paths"]["save_dir"], config["paths"]["training_load_weight"])
        net_model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"Loaded weights from {weight_path}")

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=config["training"]["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["training"]["epoch"],
                                                           eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["training"]["multiplier"],
                                             warm_epoch=config["training"]["epoch"] // 10,
                                             after_scheduler=cosineScheduler)

    return device, dataset, dataloader, net_model, optimizer, warmUpScheduler


def train_loop(config, device, dataset, dataloader, net_model, optimizer, warmUpScheduler):
    """Handles the actual training loop and checkpoint saving."""
    save_path = os.path.join(config["paths"]["save_dir"], config["paths"]["run_name"])
    checkFile(save_path)

    net_model.train()
    loss_list = []

    for e in range(config["training"]["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True,
                  desc=f"Epoch {e + 1}/{config['training']['epoch']}") as tqdmDataLoader:
            for positive, negative in tqdmDataLoader:
                # train
                combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum], axis=0)
                combined_groups = spectral_group(combined_vectors, config['model']['band'],
                                                 config['model']['group_length'])

                optimizer.zero_grad()
                x_0 = combined_groups.to(device)
                features = net_model(x_0)
                loss = isia_loss(features, config['training']['batch_size'])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net_model.parameters(), config["training"]["grad_clip"])
                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        warmUpScheduler.step()
        loss_list.append(loss.item())

        # Save only every 'save_freq' epochs, or on the very last epoch
        if (e + 1) % config['training'].get('save_freq', 1) == 0 or (e + 1) == config['training']['epoch']:
            torch.save(net_model.state_dict(), os.path.join(save_path, f"ckpt_{e + 1}_.pt"))

    paintTrend(loss_list, epochs=config['training']['epoch'], stride=5)
    # save loss list
    with open(os.path.join(save_path, "loss_list.txt"), 'w') as f:
        for loss in loss_list:
            f.write(f"{loss}\n")


def train(config: Dict):
    """Main training orchestrator."""
    device, dataset, dataloader, net_model, optimizer, warmUpScheduler = setup_training(config)
    train_loop(config, device, dataset, dataloader, net_model, optimizer, warmUpScheduler)


# ==========================================
# REFACTORED EVALUATION LOGIC
# ==========================================

def get_detection_map(model, data, map_gt, config, device):
    """Helper to extract features and compute the detection map to keep eval clean."""
    h, w, c = data.shape
    numpixel = h * w
    data_matrix = np.reshape(data, [-1, c], order='F')

    target_spectrum = ts_generation(data, map_gt, 7)
    target_prior = spectral_group(target_spectrum.T, config['model']['band'], config['model']['group_length']).to(
        device)
    target_features = model(target_prior).cpu().detach().numpy()

    detection_map = np.zeros([numpixel])
    batch_size = config['training']['batch_size']

    for i in range(0, numpixel, batch_size):
        pixels = data_matrix[i:i + batch_size]
        pixels = spectral_group(pixels, config['model']['band'], config['model']['group_length']).to(device)
        features = model(pixels).cpu().detach().numpy()
        detection_map[i:i + len(pixels)] = cosin_similarity_numpy(features, target_features)

    detection_map = np.reshape(detection_map, [h, w], order='F')
    detection_map = standard(detection_map)
    return np.clip(detection_map, 0, 1)


def load_eval_data(config):
    mat = sio.loadmat(config["paths"]["dataset_path"])
    data = standard(mat[config['data_keys']['data']])
    map_gt = mat[config['data_keys']['map']]
    return np.float32(data), map_gt


def select_best(config: Dict):
    seed_torch(config['seed'])
    device = torch.device(config["device"])
    save_path = os.path.join(config["paths"]["save_dir"], config["paths"]["run_name"])

    data, map_gt = load_eval_data(config)
    y_l = np.reshape(map_gt, [-1, 1], order='F')

    opt_epoch = 0
    max_auc = 0

    model = SpectralGroupAttention(**config['model']).to(device)

    for e in range(1, config['training']['epoch'] + 1):
        ckpt_path = os.path.join(save_path, f"ckpt_{e}_.pt")
        if not os.path.exists(ckpt_path): continue

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        detection_map = get_detection_map(model, data, map_gt, config, device)
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        auc = round(metrics.auc(fpr[1:], tpr[1:]), config['training']['epision'])

        if auc > max_auc:
            max_auc = auc
            opt_epoch = e

    print(f"Max AUC: {max_auc} at Epoch: {opt_epoch}")


def eval_model(config: Dict):
    seed_torch(config['seed'])
    device = torch.device(config["device"])

    data, map_gt = load_eval_data(config)
    model = SpectralGroupAttention(**config['model']).to(device)

    weight_path = os.path.join(config["paths"]["save_dir"], config["paths"]["run_name"],
                               config["paths"]["test_load_weight"])
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print("Model weights loaded.")
    model.eval()

    with torch.no_grad():
        detection_map = get_detection_map(model, data, map_gt, config, device)

        y_l = np.reshape(map_gt, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        auc1 = round(metrics.auc(fpr[1:], tpr[1:]), config['training']['epision'])
        auc2 = round(metrics.auc(threshold[1:], fpr[1:]), config['training']['epision'])
        auc3 = round(metrics.auc(threshold[1:], tpr[1:]), config['training']['epision'])
        auc4 = round(auc1 + auc3 - auc2, config['training']['epision'])
        auc5 = round(auc3 / auc2, config['training']['epision'])

        precision = config['training']['epision']
        for val in [auc1, auc2, auc3, auc4, auc5]:
            print(f'{val:.{precision}f}')

        plt.imshow(detection_map)
        plt.show()

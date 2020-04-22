import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import glob
from collections import OrderedDict
import multiprocessing
from torch import nn
from utils import load_model_dict
from dataset import train_loader, test_loader
import argparse
from model import AlexNet
from misc import progress_bar
from tensorboardX import SummaryWriter
from solver import Solver
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '3'


def mask_to_index(mask):
    indices = []
    for i, value in enumerate(mask):
        if value == True:
            indices.append(i)
    return indices


def get_layer_indice(model, target):
    indice = []
    if target == torch.nn.modules.conv.Conv2d:
        layers = list(model.features._modules.items())
    else:
        layers = list(model.classifier._modules.items())
    for i in layers:
        if isinstance(i[1], target):
            indice.append(int(i[0]))
    return indice


def create_solver(args, model_root='/home/huanzhang/code/new_prune/best_model_new.pkl'):
    solver = Solver(args)
    # solver.load_data()
    solver.load_model()
    model_dict = load_model_dict(model_path=model_root)
    solver.model.load_state_dict(model_dict)
    return solver


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_conv_layer(solver, layer_index, res_percent):
    true_conv_layer_index = get_layer_indice(solver.model, torch.nn.modules.conv.Conv2d)
    true_fc_layer_index = get_layer_indice(solver.model, torch.nn.modules.linear.Linear)
    last_conv_layer = true_conv_layer_index[-1]

    model_dict = solver.model.state_dict()
    _, conv = list(solver.model.features._modules.items())[true_conv_layer_index[layer_index]]
    if int(_) == last_conv_layer:
        next_conv = None
    else:
        _, next_conv = list(solver.model.features._modules.items())[true_conv_layer_index[layer_index + 1]]
    layers = list(model_dict.keys())[:10]
    feature_name, bias_name = layers[2 * layer_index], layers[2 * layer_index + 1]
    feature_layer = model_dict[feature_name].cpu().numpy()
    feature_layer = feature_layer.reshape(feature_layer.shape[0], -1)
    layer_norm = torch.norm(torch.from_numpy(feature_layer), dim=1, p=2)
    threshold = np.percentile(layer_norm, 100 - res_percent)
    mask = layer_norm >= threshold
    removed = mask.shape[0] - mask.sum().item()

    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                               out_channels=conv.out_channels - removed,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)
                               )
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = torch.index_select(torch.from_numpy(old_weights), \
                                     index=torch.Tensor(mask_to_index(mask)).long(), \
                                     dim=0).numpy()
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()
        bias = torch.index_select(torch.from_numpy(bias_numpy),
                                  index=torch.Tensor(mask_to_index(mask)).long(),
                                  dim=0).numpy()
        new_conv.bias.data = torch.from_numpy(bias).cuda()

    # Exist next conv
    if not next_conv is None:
        new_next_conv = torch.nn.Conv2d(in_channels=next_conv.in_channels - removed,
                                        out_channels=next_conv.out_channels,
                                        kernel_size=next_conv.kernel_size,
                                        stride=next_conv.stride,
                                        padding=next_conv.padding,
                                        dilation=next_conv.dilation,
                                        groups=next_conv.groups,
                                        bias=(next_conv.bias is not None))
        old_weights = next_conv.weight.data.cpu().numpy()
        new_next_conv.weight.data = torch.index_select(torch.from_numpy(old_weights),
                                                       index=torch.Tensor(mask_to_index(mask)).long(), dim=1).cuda()
        if next_conv.bias is not None:
            new_next_conv.bias.data = next_conv.bias.data
    # Strong restrain
    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(solver.model.features, i,
                             [true_conv_layer_index[layer_index], true_conv_layer_index[layer_index + 1]],
                             [new_conv, new_next_conv]) for i, _ in enumerate(solver.model.features))
        )
        del solver.model.features
        del conv
        solver.model.features = features
    else:
        # refine the fc layer
        solver.model.features = torch.nn.Sequential(
            *(replace_layers(solver.model.features, i, [true_conv_layer_index[layer_index]],
                             [new_conv]) for i, _ in enumerate(solver.model.features))
        )
        old_fc_layer = None
        for _, module in solver.model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_fc_layer = module
                break
        if old_fc_layer is None:
            raise BaseException('No Fc Linear')
        params_per_input_channel = int(old_fc_layer.in_features / conv.out_channels)

        new_fc_layer = torch.nn.Linear(old_fc_layer.in_features - params_per_input_channel * removed,
                                       old_fc_layer.out_features)
        old_weights = old_fc_layer.weight.data.cpu().numpy()
        test = new_fc_layer.weight.data.cpu().numpy()
        temp = list(map(lambda x: params_per_input_channel * x, mask_to_index(mask)))
        prune_fc_index = []
        for i, item in enumerate(temp):
            for j in range(4):
                prune_fc_index.append(item + j)

        new_weights = torch.index_select(torch.from_numpy(old_weights), index=torch.Tensor(prune_fc_index).long(),
                                         dim=1)

        # todo something
        new_fc_layer.bias.data = old_fc_layer.bias.data
        new_fc_layer.weight.data = new_weights.cuda()
        classifier = torch.nn.Sequential(
            *(replace_layers(solver.model.classifier, i, [true_fc_layer_index[0]], [new_fc_layer])
              for i, _ in enumerate(solver.model.classifier))
        )
        del solver.model.classifier
        del next_conv
        del conv
        solver.model.classifier = classifier
    # print(((layer_norm > threshold).sum()).item() / (feature_layer.shape[0]))
    return solver


def prune_fc_layer(solver, layer_index, res_percent):
    true_fc_layer_index = get_layer_indice(solver.model, torch.nn.modules.linear.Linear)
    last_fc_layer = true_fc_layer_index[-1]
    model_dict = solver.model.state_dict()
    _, fc = list(solver.model.classifier._modules.items())[true_fc_layer_index[layer_index]]
    if int(_) == last_fc_layer:
        next_fc = None
    else:
        _, next_fc = list(solver.model.classifier._modules.items())[true_fc_layer_index[layer_index + 1]]
    layers = list(model_dict.keys())[10:]
    # fc_layer = model_dict[fc_name].cpu().numpy()

    # print(type(fc_layer))
    old_weights = fc.weight.data.cpu().numpy()
    temp = torch.norm(torch.from_numpy(old_weights), p=2, dim=1)
    # todo dim?
    threshold = np.percentile(temp.numpy(), 100 - res_percent)
    mask = temp >= threshold

    removed = mask.shape[0] - mask.sum().item()
    new_fc = torch.nn.Linear(in_features=fc.in_features,
                             out_features=fc.out_features - removed,
                             )
    new_weights = torch.index_select(torch.from_numpy(old_weights),
                                     index=torch.Tensor(mask_to_index(mask)).long(), dim=0).numpy()
    new_fc.weight.data = torch.from_numpy(new_weights).cuda()
    if fc.bias is not None:
        old = fc.bias.data.cpu().numpy()

        new_np = torch.index_select(torch.from_numpy(old),
                                              index=torch.Tensor(mask_to_index(mask)).long(),
                                              dim=0).numpy()
        new_fc.bias.data = torch.from_numpy(new_np).cuda()

    # exist next fc
    if not next_fc is None:
        new_next_fc = torch.nn.Linear(in_features=next_fc.in_features - removed,
                                      out_features=next_fc.out_features)
        old_weights = next_fc.weight.data.cpu().numpy()
        new_next_fc.weight.data = torch.index_select(torch.from_numpy(old_weights),
                                                     index=torch.Tensor(mask_to_index(mask)).long(), dim=1).cuda()
        if next_fc.bias is not None:
            new_next_fc.bias.data = next_fc.bias.data
    if not next_fc is None:
        classifier = torch.nn.Sequential(
            *(replace_layers(solver.model.classifier, i,
                             [true_fc_layer_index[layer_index], true_fc_layer_index[layer_index + 1]],
                             [new_fc, new_next_fc]) for i, _ in enumerate(solver.model.classifier))

        )
        # print(new_fc.weight.data.shape)
        # print(new_next_fc.weight.data.shape)
        # print(new_fc.bias.data.shape)
        # print(new_next_fc.bias.data.shape)
        del solver.model.classifier
        del fc
        solver.model.classifier = classifier
    else:
        raise BaseException("last layer cannot be pruned")
    return solver


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()
    # solver = create_solver(args=args)
    # solver.load_data()
    # conv_indices = get_layer_indice(solver.model, torch.nn.modules.conv.Conv2d)
    # fc_indices = get_layer_indice(solver.model, torch.nn.modules.linear.Linear)

    #x = list(np.logspace(2, 0, num=100))
    x = list(np.logspace(0,-2.522878745280338,num=100))
    y = list(np.logspace(1.301029995663981, -2.522878745280338, num=100))
    print(y)
    coord = []
    acc = []
    run_time = []
    # solver = prune_fc_layer(create_solver(args=args), 0, 20)
    # solver = prune_fc_layer(solver, 1,70)
    # print(solver.model)
    # print(solver.model)
    solver_pre = create_solver(args=args)
    solver_pre.load_data()
    # for layer in [0, 1]:
    #     for i in x:
    #         solver = create_solver(args=args)
    #         solver.train_loader = solver_pre.train_loader
    #         solver.test_loader = solver_pre.test_loader
    #         solver = prune_fc_layer(solver, layer, i)
    #         _, acc_temp, run_time_temp = solver.test()
    #         acc.append(acc_temp)
    #         run_time.append(run_time_temp)
    #         coord.append(i)
    #
    #     with open('./fc/fc:' + str(layer) + '% weight_res.txt','w+') as f:
    #         f.write(str(coord))
    #     with open('./fc/fc:' + str(layer) + '% acc.txt', 'w+') as f:
    #         f.write(str(acc))
    #     with open('./fc/fc:' + str(layer) + '% run_time.txt','w+') as f:
    #         f.write(str(run_time))
    layer = 0
    for i in y:
        solver = create_solver(args=args)
        solver.test_loader = solver_pre.test_loader
        solver = prune_fc_layer(solver, layer, i)
        _, acc_temp, run_time_temp = solver.test()
        print(acc_temp,run_time_temp,i)
        acc.append(acc_temp)
        run_time.append(run_time_temp)
        coord.append(i)

    with open('./fc/fc:' + str(layer) + '% weight_res.txt', 'w+') as f:
        f.write(str(coord))
    with open('./fc/fc:' + str(layer) + '% acc.txt', 'w+') as f:
        f.write(str(acc))
    with open('./fc/fc:' + str(layer) + '% run_time.txt', 'w+') as f:
        f.write(str(run_time))

    # for pre_percent in x:
    #     for percent in y:
    #         sovler = create_solver(args=args)
    #         #sovler.load_data()
    #         sovler = prune_conv_layer(sovler, 0, 70)
    #         sovler = prune_conv_layer(sovler, 1, 80)
    #         break
    #         print('layer 2' + ', ' + str(pre_percent) + 'res, layer 3' + str(percent) + 'res')
    #         _, acc_temp, run_time_temp = sovler.test()
    #         coord.append([pre_percent, percent])
    #         acc.append(acc_temp)
    #         run_time.append(run_time_temp)
    # #2,3
    # with open('./coord_res_01.txt', 'w+') as f:
    #     f.write(str(coord))
    # with open('./acc_01.txt', 'w+') as f:
    #     f.write(str(acc))
    # with open('./run_time_01.txt', 'w+') as f:
    #     f.write(str(run_time))

    # solver = create_solver(args=args)
    # solver.load_data()
    # solver = prune_conv_layer(solver, i, percent)
    # print('layer ' + str(i) + ', ' + str(percent) + 'res')
    # _, acc_temp, run_time_temp = solver.test()
    # per.append(percent)
    # run_time.append(run_time_temp)
    # acc.append(acc_temp)
    #
    # with open('./feature' + str(i) + ' :percent_res.txt', 'w+') as f:
    #     f.write(str(per))
    # with open('./feature' + str(i) + ' :acc.txt', 'w+') as f:
    #     f.write(str(acc))
    # with open('./feature' + str(i) + ' :run_time.txt', 'w+') as f:
    #     f.write(str(run_time))
    # per.clear()
    # acc.clear()
    # run_time.clear()
    # for i in range(2):
    #     prune_fc_layer(create_solver(args=args),i,100)

    # model_dict = load_model_dict(model_path='/home/huanzhang/code/new_prune/best_model_new.pkl')
    # solver.model.load_state_dict(model_dict)
    # solver.load_data()
    # model_dict = solver.model.state_dict()
    # for k, v in model_dict.items():
    #     print(v.shape)
    # _,acc = solver.test()
    # print(type(acc))
    # print(solver.model)


if __name__ == '__main__':
    main()

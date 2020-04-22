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
import pickle

os.environ['CUDA_VISIBLE_DEVICE'] = '2'


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


def get_layers_name(solver):
    layers = list(solver.model.state_dict().keys())[: 10]
    return layers


def prune_conv_layer(solver, layer_index, res_percent, layers):
    true_conv_layer_index = get_layer_indice(solver.model, torch.nn.modules.conv.Conv2d)
    true_fc_layer_index = get_layer_indice(solver.model, torch.nn.modules.linear.Linear)
    last_conv_layer = true_conv_layer_index[-1]

    model_dict = solver.model.state_dict()
    _, conv = list(solver.model.features._modules.items())[true_conv_layer_index[layer_index]]
    if int(_) == last_conv_layer:
        next_conv = None
    else:
        _, next_conv = list(solver.model.features._modules.items())[true_conv_layer_index[layer_index + 1]]
    # layers = list(model_dict.keys())[:10]
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
                               # dilation=conv.dilation,
                               # groups=conv.groups,
                               # bias=(conv.bias is not None)
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
        del solver.model.classifier
        del fc
        solver.model.classifier = classifier
    else:
        raise BaseException("last layer cannot be pruned")
    return solver


class Reusult:
    def __init__(self, init_acc, init_runtime, init_res, init_param):
        self.acc = init_acc
        self.runtime = init_runtime
        self.weight_res = init_res
        self.param = init_param


def cal_para(solver):
    return sum(param.numel() for param in solver.model.parameters())


def prune_all_conv_layers(args):
    f0 = list(np.linspace(60, 85, num=7))[::-1]
    f1 = list(np.linspace(25, 35, num=5))[::-1]
    f2 = list(np.linspace(25, 45, num=7))[::-1]
    f3 = list(np.linspace(35, 45, num=5))[::-1]
    f4 = list(np.linspace(5, 25, num=7))[::-1]
    pre_solver = create_solver(args=args)
    pre_solver.load_data()
    _, init_acc, init_runtime = pre_solver.test()

    init_res = []
    for i in [f0, f1, f2, f3, f4]:
        init_res.append(i[0])
    min_paras = cal_para(pre_solver)
    result = Reusult(init_acc, init_runtime, init_res, min_paras)
    delta = 0.05
    strict_delta = 0.04
    layers = get_layers_name(pre_solver)
    strict_num = 0
    for x0 in f0:
        for x1 in f1:
            for x2 in f2:
                for x3 in f3:
                    for x4 in f4:
                        print(x0, x1, x2, x3, x4)
                        solver = create_solver(args=args)
                        solver.test_loader = pre_solver.test_loader
                        weight_res = [x0, x1, x2, x3, x4]
                        for i, value in enumerate(weight_res):
                            solver = prune_conv_layer(solver, i, value, layers)
                        _, acc_temp, run_time_temp = solver.test()
                        param = cal_para(solver)
                        if acc_temp >= init_acc - strict_delta:
                            with open('./result/result' + str(strict_num) + '.pkl', 'wb+') as f:
                                pickle_result = Reusult(acc_temp, run_time_temp, weight_res, param)
                                strict_num = strict_num + 1
                                pickle.dump(pickle_result, f)
                        if acc_temp >= init_acc - delta and min_paras >= param:
                            min_paras = param
                            result.acc = acc_temp
                            result.runtime = run_time_temp
                            result.weight_res = weight_res
                            result.param = param

    with open('./result.txt', 'w+') as f:
        f.write(str(result.weight_res) + '\n')
        f.write(str(result.runtime) + '\n')
        f.write(str(result.acc) + '\n')
        f.write(str(result.param) + '\n')


def prune_all_fc_layers(args):
    pre_solver = create_solver(args)
    pre_solver.load_data()
    coord = []
    acc = []
    run_time = []
    param = []
    x = np.linspace(14,0.0025,num=100)
    y = np.linspace(5,0.0025,num=100)
    for fc0 in x:
        for fc1 in y:
            solver = create_solver(args)
            solver.test_loader = pre_solver.test_loader
            solver = prune_fc_layer(solver,0,fc0)
            solver = prune_fc_layer(solver,1,fc1)
            _, acc_temp, run_time_temp = solver.test()
            coord.append([fc0,fc1])
            acc.append(acc_temp)
            run_time.append(run_time_temp)
            param.append(cal_para(solver))

    with open('./coord_fc.txt', 'w+') as f:
        f.write(str(coord))
    with open('./acc_fc.txt', 'w+') as f:
        f.write(str(acc))
    with open('./run_time_fc.txt', 'w+') as f:
        f.write(str(run_time))
    with open('./params_fc.txt') as f:
        f.write(str(param))


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()
    # fc 2layer
#    prune_all_fc_layers(args)
    solver = Solver(args)
    solver.load_data()
    solver.model = AlexNet()
    _, acc, _ = solver.test()
    print(acc)
    #feature 5 layers
    # prune_all_conv_layers(args)

    # sovler.load_data()
    # sovler = prune_conv_layer(sovler, 0, 70, get_layers_name(create_solver(args=args)))
    # sovler = prune_conv_layer(sovler, 1, 80, get_layers_name(create_solver(args=args)))

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


if __name__ == '__main__':
    main()

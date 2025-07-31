import numpy as np
import torch


def compute_local_test_accuracy(model, dataloader, data_distribution):
    model.eval()
    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                total_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()

    # model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def compute_acc(net, test_data_loader):
    net.cuda()
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    # net.to('cpu')
    return correct / float(total)


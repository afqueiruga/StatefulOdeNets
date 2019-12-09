import numpy as np
import torch
import torch.nn.functional as F




def test_ori(model, test_loader, num_data):
    model.eval()
    correct = 0
    total_ent = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            ent = F.softmax(output, dim=0)
            tmp_A = sum(ent * torch.log(ent+1e-6))
            total_ent += tmp_A[0]
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    print('correct: ', correct / num_data)
    return correct/num_data, total_ent/num_data
            




def test_adv(adv_data, Y_test, model, num_data):
    num_iter = num_data // 100
    model.eval()
    correct = 0
    total_ent = 0.
    with torch.no_grad():
        for i in np.arange(num_iter):
            data, target = adv_data[100*i:100*(i+1), :], Y_test[100*i:100*(i+1)]
            data, target = data.cuda(), target.cuda()
            output = model(data)
            ent = F.softmax(output, dim=0)
            tmp_A = sum(ent * torch.log(ent+1e-6))
            total_ent += tmp_A[0]
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    return correct*1./num_data, total_ent*1./num_data




def distance(X_adv, X_prev, norm=2):
    n = len(X_adv)
    dis = 0.0
    dis_abs = 0.0
    large_dis = 0.0
    
    for i in range(n):
        if norm == 2:
            tmp_dis_abs = torch.norm(X_adv[i,:] - X_prev[i,:], p=norm)
            tmp_dis = tmp_dis_abs / torch.norm(X_prev[i,:], p=norm)
        if norm == 1:
            tmp_dis_abs = torch.max(torch.abs(X_adv[i,:] - X_prev[i,:]))
            tmp_dis = tmp_dis_abs / torch.max(torch.abs(X_prev[i,:]))        
        
        dis += tmp_dis
        dis_abs += tmp_dis_abs
        large_dis = max(large_dis, tmp_dis)
        
    return dis/n, dis_abs/n, large_dis

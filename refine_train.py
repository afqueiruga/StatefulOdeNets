import torch
from odenet import refine

#
# helper functions to examine models
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_for_epochs(model, loader,
                     criterion,
                     N_epochs, losses = None, lr=1.0e-3,N_print=1000):
    "Works for normal models too"
    if losses is None:
        losses = []
    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_count = 0
    for e in range(N_epochs):
        for imgs,labels in iter(loader):
            optimizer.zero_grad()
            out = model(imgs)
            L = criterion(out,labels)
            L.backward()
            optimizer.step()
            losses.append(L.detach().numpy())
            if step_count % N_print == N_print-1:
                print(L.detach())
                plot_pred(model)
                plt.show()
            step_count += 1
    return losses


def train_adapt(model, loader, criterion, N_epochs, N_refine):
    """I don't know how to control the learning rate"""
    losses = []
    refine_steps = []
    model_list = [model]
    N_print= 10000
    for i in range(N_refine):
        lr =  10.0**(-1-i//2)
        if i > 0:
            model_list.append(model_list[-1].refine())
            print("Adapting to ", count_parameters(model_list[-1]), "with lr = ",lr)
        else:
            print("Starting with ",count_parameters(model_list[-1]), "with lr = ",lr)
        losses = train_for_epochs(model_list[-1],loader, criterion,N_epochs,losses, lr = lr)
        refine_steps.append(len(losses))
    return model_list, losses, refine_steps


def plot_weights_over_time(model_list, grab_it):
    for i,m in enumerate(model_list):
        w = grab_it(m).detach().numpy()
        ts =  m.net[1].ts.detach().numpy()
        #print(ts.shape, w.shape)
        #plt.imshow(w[:,0,:].T)
        plt.subplot(len(model_list),1,i+1)
        dt = ts[1]-ts[0]
        plt.bar(ts[0:-1]+dt*0.5,w,width=ts[1]-ts[0],edgecolor='k')
        plt.xlabel('t')
    plt.show()
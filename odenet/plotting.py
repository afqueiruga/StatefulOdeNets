from matplotlib import pylab as plt
import torch

#
# Evaluation tools
#
def acc(y,labels):
    return torch.sum(torch.argmax(y,dim=-1) == labels)*1.0/len(labels)
def model_acc(model,loader):
    try:
        imgs,labels = loader
    except:
        imgs,labels = next(iter(loader))
    y = model(imgs.to(which_device(model)))
    return acc(y.cpu(),labels)


#
# Plotting tools
#
def plot_accuracy(model,loader):
    try:
        imgs,labels = loader
    except:
        imgs,labels = next(iter(loader))
    y = model(imgs.to(which_device(model)))
    print(acc(y.cpu(),labels).item())
    bars = torch.nn.Softmax(dim=-1)(y[:10])
    size = len(bars)
    plt.figure(figsize=(10,10))
    for i,(pred,img,label) in enumerate(zip(bars,imgs,labels)):
        plt.subplot(size//2+1,4,1+2*i)
        plt.imshow(img[0,:,:].detach().numpy(),cmap='Greys')
        plt.subplot(size//2+1,4,2+2*i)
        plt.bar(range(10),[1 if y==label else 0 for y in range(10)])
        plt.bar(range(10),pred.cpu().detach().numpy())
    plt.show()

def plot_weights_over_time(model_list, grab_w, grab_ts):
    for i,m in enumerate(model_list):
        w = grab_w(m).detach().numpy()
        ts =  grab_ts(m).detach().numpy()
        #print(ts.shape, w.shape)
        #plt.imshow(w[:,0,:].T)
        plt.subplot(len(model_list),1,i+1)
        dt = ts[1]-ts[0]
        plt.bar(ts[0:-1]+dt*0.5,w,width=ts[1]-ts[0],edgecolor='k')
        plt.xlabel('t')
    plt.show()

def plot_layers_over_times(model, img):
    y = models.channel_squish(img,2)
    with torch.no_grad():
        yy = torchdiffeq.odeint(m.net[0].net , y, m.net[0].ts)
    plt.figure(figsize=(8,20))
    L = yy.shape[0]
    for i in range(L):
        for j in range(4):
            plt.subplot(L,4,4*i+j+1)
            plt.imshow(yy[i,2,j,:,:])
    plt.show()
   
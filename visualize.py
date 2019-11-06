
from preporcess.CrowdaiData import *
import yaml
import torch
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def visualize(model, val_loader):
    model.eval()
    plt.figure()
    with torch.no_grad():
        for batch_i,(image,target) in enumerate(val_loader):
            modelimage=image.to(device)
            output=model(modelimage)
            for i in range(image.shape[0]):
                timage=np.moveaxis(image[i].numpy(),0,-1)
                tmask=output[i].cpu().numpy()[0,:,:]
                ttarget=target[i].cpu().numpy()[0,:,:]
                plt.subplot(1, 2, 1*(i+1))
                plt.imshow(timage)
                plt.subplot(1, 2, 1*(i+1)+1)
                plt.imshow(np.array(tmask>0.5,np.float))
                inter=np.sum(np.logical_and(ttarget,np.array(tmask>0.5,np.float)))
                union = np.sum(np.logical_or(ttarget, np.array(tmask > 0.5, np.float)))
                print(1.0*inter/union)
                break
            break
    plt.show()
from main import modeldict
if __name__=='__main__':
    config_path='config.yml'
    weightpath='/Disk4/xkp/project/BuildingSeg/LOG/Unet8/Iout-0.7944.pkl'
    f=open(config_path)
    config=yaml.load(f)
    traindataloader,valdaraloader=GetDataloader(trainimagepath=config['CrowdaiData']['trainimagepath'],
                                                 trainmaskpath=config['CrowdaiData']['trainmaskpath'],
                                                  valimagepath=config['CrowdaiData']['valimagepath'],
                                                   valmaskpath=config['CrowdaiData']['valmaskpath'],
                                                shape=config['CrowdaiData']['shape'],
                                                padshape=config['CrowdaiData']['padshape'],batchsize=2,
                                                numworkers=int(cpu_count()))

    model=modeldict[config['modeltype']](50,3).to(device)
    model.load_state_dict(torch.load(weightpath))
    visualize(model,valdaraloader)



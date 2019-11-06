from torch import nn
import yaml
import logging
import time
from models.unet import UNet8
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch import optim
from utils.losses import get_lossf
import socket
print(socket.gethostname())
from utils.metrics import *
from preporcess.CrowdaiData import GetDataloader
from multiprocessing import cpu_count
modeldict={
    'Unet8':UNet8
}
import os
from utils.RAdam import RAdam
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(socket.gethostname()+"log.txt")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(model, val_loader,epoch=0):
    model.eval()
    Ioul=[]
    Ioutl=[]
    with torch.no_grad():
        for batch_i,(image, yt), in enumerate(val_loader):
            image, yt= image.to(device),yt.to(device)
            yp= model(image)
            ypl=[]
            ytl=[]
            for o in yp.cpu():
                ypl.append(o)
            for y in yt.cpu():
                ytl.append(y)
            Iou,IoUt=IoU(ypl,ytl),iout(ypl,ytl)
            Ioul.extend(Iou)
            Ioutl.extend(IoUt)
            if batch_i>1000:
                break
    return np.mean(Ioul),np.mean(Ioutl)



def train(config_path):
    logger.info("-----------Start parse experiments-------------")
    f=open(config_path)
    config=yaml.load(f)

    logger.info(str(device) + ' Device is available ')

    traindataloader,valdaraloader=GetDataloader(trainimagepath=config['CrowdaiData']['trainimagepath'],
                                                 trainmaskpath=config['CrowdaiData']['trainmaskpath'],
                                                  valimagepath=config['CrowdaiData']['valimagepath'],
                                                   valmaskpath=config['CrowdaiData']['valmaskpath'],
                                                shape=config['CrowdaiData']['shape'],
                                                padshape=config['CrowdaiData']['padshape'],batchsize=config['batchsize'],
                                                numworkers=int(cpu_count()))
    logger.info("Obtain Dataloader successful ")
    pathdir=os.path.join(config['dataroot'],config['modellogdir'],config['modeltype'])
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)
    logger.info('The model ckp will save in :'+' <  '+pathdir+'  >')
    model=modeldict[config['modeltype']](50,3).to(device)
    bestiout=0.0
    if not config['init_ckp'] == 'None':
        CKP=config['init_ckp']
        logger.info('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    else:
        logger.info('Not loadding weights')

    if config['optimizer']=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=0.0001)
    else:
        # optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.0001)
        optimizer=RAdam(model.parameters(), lr=config['lr'], weight_decay=0.0001)
    logger.info('Using '+config['optimizer']+' as optimizer ')
    if config['lr_scheduler'] == 'RP':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, min_lr=0.00001)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, 15, eta_min=0.00001)
    logger.info('Using ' + config['lr_scheduler'] + ' as  learning scheduler')

    model.train()
    lossfunction=get_lossf()
    lr_scheduler.step()
    for epoch in range(config['epoch']):
        train_loss=0.0
        clr=get_lrs(optimizer)
        bg=time.time()
        model.train()
        print('epoch |   lr    |         %         |f1score |  loss  |  avg   |  iou   | iout   |  best  | time |  filepath   |')
        for batch_i, (imgs, targets) in enumerate(traindataloader):
            imgs=imgs.to(device)
            targets=targets.to(device)
#            print(imgs.cpu().detach().numpy().max(),imgs.cpu().detach().numpy().min(),targets.cpu().detach().numpy().min(),targets.cpu().detach().numpy().max())
            optimizer.zero_grad()
            output=model(imgs)
            loss=lossfunction(output, targets)
            F1SCORE=F1score(output.cpu().detach(),targets.cpu().detach())
            print('\r {:4d} | {:.5f} | {:8d}/{:8d} | {:.4f} | {:.4f} | {:.4f} |'.format(
                epoch, float(clr[0]), config['batchsize'] * (batch_i + 1), traindataloader.__len__()*config['batchsize'], F1SCORE,loss.item(),
                                             train_loss / (batch_i + 1)), end='')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_i>280741//8:
                break
        iou,iout=validate(model,valdaraloader,epoch)
        if iout>bestiout:
            bestiout = iout
            path=pathdir+'/'+'Iout-{:.4f}-IoU-{:.4f}.pkl'.format(bestiout,iou)
            torch.save(model.state_dict(),path)
        print(' {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s}                               |'.format(iou, iout, bestiout, (time.time() - bg) / 60,pathdir))
        if config['lr_scheduler']== 'RP':
            lr_scheduler.step(bestiout)
        else:
            lr_scheduler.step()
            


    del model,traindataloader,valdaraloader,optimizer,lr_scheduler
    torch.cuda.empty_cache()
    logger.info('-------Experiment have finish!-------')
if __name__=='__main__':
    train('config.yml')




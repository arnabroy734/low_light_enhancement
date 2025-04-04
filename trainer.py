from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader
from DCENet import DCENet, ColorConsistencyLoss, SpatialConsistencyLoss, IlluminationSmoothnessLoss, ExposureLoss
import pickle
import torch
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 ds_name,
                 param,
                 modelpath,
                 epochs=10,
                 lr=10**(-4),
                 batch=8
            ):
        trainds, valds = get_dataset(ds_name)
        self.trainloader = DataLoader(trainds, batch_size=batch)
        self.valloader = DataLoader(valds, batch_size=batch)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.get_param(modelpath) is None:
            self.param = param
        else:
            self.param = self.get_param(modelpath)
        self.model = self.get_model(modelpath)
        self.model.to(self.device)
        self.losses = self.get_losses(modelpath)
        self.epochs = epochs
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.exp_loss_fn = ExposureLoss()
        self.cc_loss_fn = ColorConsistencyLoss()
        self.illsm_loss_fn = IlluminationSmoothnessLoss()
        self.spcon_loss_fn = SpatialConsistencyLoss(self.device)
        self.modelpath = modelpath
    
    def forward_step(self, x_batch, validation=False):
        if validation is True:
            with torch.no_grad():
                alphas, X_recons = self.model.generate(x_batch)
        else:
            alphas, X_recons = self.model.generate(x_batch)
        exp_loss = self.param['exp_w']*self.exp_loss_fn(X_recons)
        cc_loss = self.param['cc_w']*self.cc_loss_fn(X_recons)
        illsm_loss = self.param['illsm_w']*self.illsm_loss_fn(alphas)
        spcon_loss = self.param['spcon_w']*self.spcon_loss_fn(X_recons, x_batch)
        return exp_loss, cc_loss, illsm_loss, spcon_loss
    
    def train(self):
        for i in range(self.epochs):
            l_cc_tr = list()
            l_exp_tr = list()
            l_spcon_tr = list()
            l_illsm_tr = list()

            l_cc_val = list()
            l_exp_val = list()
            l_spcon_val = list()
            l_illsm_val = list()

            for j, x_batch in enumerate(self.trainloader):
                x_batch = x_batch.to(self.device)
                exp_loss, cc_loss, illsm_loss, spcon_loss = self.forward_step(x_batch)
                loss = exp_loss +  cc_loss +  illsm_loss +  spcon_loss
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                l_cc_tr.append(cc_loss.item())
                l_exp_tr.append(exp_loss.item())
                l_spcon_tr.append(spcon_loss.item())
                l_illsm_tr.append(illsm_loss.item())
                if j%20 == 0:
                    print (f"Epoch {i+1} | step {j+1} | total loss - {loss.item()}")

            l_cc_tr = np.mean(l_cc_tr)
            l_exp_tr = np.mean(l_exp_tr)
            l_spcon_tr = np.mean(l_spcon_tr)
            l_illsm_tr = np.mean(l_illsm_tr)
            self.losses['training']['l_cc'].append(l_cc_tr)
            self.losses['training']['l_exp'].append(l_exp_tr)
            self.losses['training']['l_spcon'].append(l_spcon_tr)
            self.losses['training']['l_illsm'].append(l_illsm_tr)
            print(f'Epoch {i+1} train | exposure- {l_exp_tr}, colconsi - {l_cc_tr}, ill smoo - {l_illsm_tr}, spcon - {l_spcon_tr}')
            
            for j, x_batch in enumerate(self.valloader):
                x_batch = x_batch.to(self.device)
                exp_loss, cc_loss, illsm_loss, spcon_loss = self.forward_step(x_batch, validation=True)
                l_cc_val.append(cc_loss.item())
                l_exp_val.append(exp_loss.item())
                l_spcon_val.append(spcon_loss.item())
                l_illsm_val.append(illsm_loss.item())
            l_cc_val = np.mean(l_cc_val)
            l_exp_val = np.mean(l_exp_val)
            l_spcon_val = np.mean(l_spcon_val)
            l_illsm_val = np.mean(l_illsm_val)
            self.losses['validation']['l_cc'].append(l_cc_val)
            self.losses['validation']['l_exp'].append(l_exp_val)
            self.losses['validation']['l_spcon'].append(l_spcon_val)
            self.losses['validation']['l_illsm'].append(l_illsm_val)
            print(f'Epoch {i+1} validation | exposure- {l_exp_val}, colconsi - {l_cc_val}, ill smoo - {l_illsm_val}, spcon - {l_spcon_val}')

        self.save_result(self.modelpath)

    def get_param(self, path):
        try:
            with open(path/'param.json', 'r') as f:
                param = json.load(f)
                f.close()
            print(f'Saved params loaded successfully')
            return param
        except Exception as e:
            return None

    def get_model(self, path):
        if path.exists() == False:
            path.mkdir(parents=True)
        try:
            with open(path/'model.pkl', 'rb') as f:
                model = pickle.load(f)
                f.close()
                print('Saved model loaded successfully')
        except Exception as e:
            model = DCENet()
        return model

    def get_losses(self,path):
        try:
            with open(path/'loss.json', 'r') as f:
                losses = json.load(f)
                f.close()
                print('Saved losses loaded successfully')
        except Exception as e:
            losses = {
                        'training': {
                            'l_cc':[], 'l_exp':[], 'l_spcon':[], 'l_illsm':[]
                    }, 
                        'validation' : {
                            'l_cc':[], 'l_exp':[], 'l_spcon':[], 'l_illsm':[]
                    }
                }
        return losses

    def save_result(self, path):
        tot_epochs = len(self.losses['training']['l_cc'])
        with open(path/f'model_{tot_epochs}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
            f.close()
        with open(path/f'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
            f.close()
        with open(path/'loss.json', 'w') as f:
            json.dump(self.losses, f)
            f.close()
        with open(path/'param.json', 'w') as f:
            json.dump(self.param, f)
            f.close()

def plot_loss_curve(path):
    with open(path/'loss.json', 'r') as f:
        losses = json.load(f)
        f.close()
    for key in losses['training'].keys():
        plt.plot(losses['training'][key], label=f'{str(key)}_training')
        plt.plot(losses['validation'][key], label=f'{str(key)}_validation')
        plt.savefig(path/f'{str(key)}_loss.png')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Experiment 7
    # param = {'exp_w':20, 'cc_w':100, 'illsm_w': 2000, 'spcon_w': 10}
    # trainer = Trainer(
    #     'lol',
    #     param,
    #     Path.cwd()/'models'/'experiment_7',
    #     epochs=10,
    #     lr=10**(-4),
    #     batch=8
    # )
    # trainer.train()
    plot_loss_curve(Path.cwd()/'models'/'experiment_7')


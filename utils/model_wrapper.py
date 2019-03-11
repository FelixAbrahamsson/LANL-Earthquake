## Imports
import torch
import numpy as np
from torch.optim import lr_scheduler
from utils.models import *
import pandas as pd
import os
import copy
import warnings


class ModelWrapper:
    
    def __init__(self, config=None, pretrained_path=None):
        
        if config is not None:
            self.config = config.copy()
        if pretrained_path is not None:
            self.config = torch.load(pretrained_path + '_config.pth')

        self.config['use_cuda'] = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.config['use_cuda'] else "cpu")
        
        self.init_model(pretrained_path=pretrained_path)
        self.init_optimizer()
        
    def init_model(self, pretrained_path=None):
        
        model_choice = self.config['model_choice']
    
        if model_choice == 0:
            self.net = ConvTransformer(self.config).to(self.device)
            self.weight_initialization(init='kaiming')

        elif model_choice == 1:
            self.net = LSTMNet(self.config).to(self.device)
            self.weight_initialization(init='kaiming')

        elif model_choice == 2:
            self.net = FFNet(self.config).to(self.device)
            
        else:
            print("ERROR: NO MODEL SELECTED")
            return False

        if pretrained_path is not None:
            self.load_state(pretrained_path)
        
        return True
    
    def init_optimizer(self):

        optim_choice = self.config['optim_choice']

        if optim_choice == 0:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'])
            self.lr_scheduler = None
            
        elif optim_choice == 1:
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.00001, momentum=0.9)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        else:
            print("ERROR: NO OPTIMIZER SELECTED")
            return False
            
        return True
    
    def weight_initialization(self, init='xavier', keywords=['']):
        for name, param in self.net.named_parameters():
            if 'weight' in name and any(k in name for k in keywords) \
                                and len(param.shape) > 1:
                if init == 'xavier':
                    torch.nn.init.xavier_normal_(param)
                elif init == 'kaiming':
                    torch.nn.init.kaiming_normal_(param)


    def get_summary(self):

        param_counts = [[n, p.numel()] for n, p in self.net.named_parameters()]
        params_summary = pd.DataFrame(param_counts, columns=['name', '# params'])
        num_params = params_summary['# params'].sum()
        params_summary['# params'] = list(map("{:,}".format, params_summary['# params']))

        return params_summary, num_params

    def update_config(self, changes):

        for param in changes:
            if param not in self.config:
                warning_str = "'{}' not in config.".format(param)
                warnings.warn(warning_str)
            
        self.config.update(changes)

        if 'lr' in changes:
            self.update_learning_rate(changes['lr'])

    def update_learning_rate(self, lr):

        self.config['lr'] = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train_loader, val_loader, verbose=2):
    
        self.net.train()

        if self.config['revert_after_training']:
            best_loss = self.evaluate(val_loader)
        else:
            best_loss = np.inf

        best_params = copy.deepcopy(self.net.state_dict())
        tot_step = 0
        smooth_loss = 0.0
        best_step = 1 # Dont stop before this many steps + patience

        eval_step = int(self.config['eval_step'] * len(train_loader))
        patience = int(self.config['patience'] * eval_step)

        for e in range(1, self.config['num_epochs'] + 1):

            if patience and tot_step > best_step + patience and best_loss != np.inf:
                break
                    
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            if verbose >= 1:
                print("---------- EPOCH {}/{} ----------\n".format(e, 
                                            self.config['num_epochs']))

            for i, batch in enumerate(train_loader):

                if patience and tot_step > best_step + patience and best_loss != np.inf:
                    ## Breaking here makes pytorch sad
                    continue

                ## Do forward pass
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                output, loss = self.net(features, labels=labels)

                if tot_step == 0:
                    smooth_loss = loss.item()
                else:
                    smooth_loss = smooth_loss * 0.99 + loss.item()*0.01

                ## Do backward pass
                self.optimizer.zero_grad()
                loss.backward()
                if self.config['clip']:
                    nn.utils.clip_grad_norm_(self.net.parameters(), 
                                             self.config['clip'])
                self.optimizer.step()

                tot_step += 1

                if tot_step % eval_step == 0 and (not patience or tot_step > patience or best_loss != np.inf):

                    val_loss = self.evaluate(val_loader)

                    if val_loss < best_loss:
                        best_params = copy.deepcopy(self.net.state_dict())
                        best_loss = val_loss
                        best_step = tot_step
                        if verbose >= 2:
                            print("New best!")
                    
                    if verbose >= 2:
                        print(
                            "Step: {}/{}\n".format(i+1, len(train_loader)) +
                            "Total steps: {}\n".format(tot_step) +
                            "Training Loss (smooth): {:.3f}\n".format(smooth_loss) +
                            "Validation Loss: {:.3f}".format(val_loss)
                        )

                        if self.config['use_cuda']:
                            max_allocated = torch.cuda.max_memory_allocated() / 10**9
                            print("Maximum GPU consumption so far: {} [GB]".format(round(max_allocated, 3)))
                            
                        print()
                    
        if verbose >= 1:
            print("Best validation loss:", best_loss)
            print("At step:", best_step+1)
            
        if self.config['revert_after_training']:
            self.net.load_state_dict(best_params) ## Revert parameters to best step
        
        return best_loss

    def evaluate(self, loader):

        loss = 0.0
        tot_sequences = 0

        self.net.eval() # Put models in evaluation mode

        with torch.no_grad():

            for batch in loader:

                ## Get data
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                batch_size = len(features)
                tot_sequences += batch_size

                ## Get output
                output, batch_loss = self.net(features, labels=labels)
                loss += batch_loss.item() * batch_size

        self.net.train() # Put model back in training mode

        return loss / tot_sequences

    def predict(self, loader):

        self.net.eval() # Put model in evaluation mode
        preds = []
        seg_ids = []

        with torch.no_grad():

            for batch in loader:

                ## Get data
                features = batch['features'].to(self.device)

                ## Get output
                output = self.net(features)

                batch_preds = output.cpu().numpy()
                preds.extend(batch_preds)

                if 'seg_id' in batch:
                    seg_ids.extend(batch['seg_id'])

        if seg_ids:
            return np.array(preds).squeeze(1), seg_ids
        else:
            return np.array(preds)

    def save_state(self, folder, name):

        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.net.state_dict(), folder + name + '.pth')
        torch.save(self.config, folder + name + '_config.pth')

    def load_state(self, path):

        self.config = torch.load(path + '_config.pth')
        self.config['use_cuda'] = torch.cuda.is_available()
        
        if self.config['use_cuda']:
            state_dict = torch.load(path + '.pth')
        else:
            state_dict = torch.load(path + '.pth', map_location='cpu')

        self.net.load_state_dict(state_dict)

        
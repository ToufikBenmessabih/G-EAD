import importlib
import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils import dotdict
from utils import calculate_mof
from postprocess import PostProcess
import torch.nn.functional as F
from dataset_inHard_13 import AugmentDataset, collate_fn_override
import torch
from torch.optim.lr_scheduler import _LRScheduler
torch.autograd.set_detect_anomaly(True)

seed = 42


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--dataset_name', type=str, default="breakfast", choices=['InHARD_13', 'IKEA-ASM'])
my_parser.add_argument('--split', type=int, required=True, help="Split number of the dataset")
my_parser.add_argument('--cudad', type=str, default='0', help="Cuda device number to run the program")
my_parser.add_argument('--base_dir', type=str, help="Base directory containing groundTruth, features, splits, results directory of dataset")
my_parser.add_argument('--model_path', type=str, default='model_5_ed_infoGcn_b_v2')
my_parser.add_argument('--wd', type=float, required=False, help="Provide weigth decay if you want to change from default")
my_parser.add_argument('--lr', type=float, required=False, help="Provide learning rate if you want to change from default")
my_parser.add_argument('--chunk_size', type=int, required=False, help="Provide chunk size to be used if you want to change from default")
my_parser.add_argument('--ensem_weights', type=str, required=False,
                        help='Default = \"1,1,1,1,0,0\", provide in similar comma-seperated 6 weights values if required to be changed')
my_parser.add_argument('--ft_file', type=str, required=False, help="Provide feature file dir path if default is not base_dir/features")
my_parser.add_argument('--ft_size', type=int, required=False, help="Default=2048 for I3D features, change if feature size changes")
my_parser.add_argument('--err_bar', type=int, required=False)
my_parser.add_argument('--num_workers', type=int, default=0, help="Number of workers to be used for data loading")
my_parser.add_argument('--out_dir', required=False, help="Directory where output(checkpoints, logs, results) is to be dumped")
args = my_parser.parse_args()


if args.err_bar:
    seed = args.err_bar #np.random.randint(0, 999999)

# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
set_seed()

# Device configuration
os.environ['CUDA_VISIBLE_DEVICES']=args.cudad
print('is cuda available ?: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

#device = torch.device("cpu")

config = dotdict(
    epochs = 500,
    dataset = args.dataset_name,
    #feature_size = 75, # shape (None, 25, 3)
    feature_size = 63, # shape (None, 21, 3)
    #feature_size = 216, # shape (None, 72, 3)
    split_number = args.split,
    model_path = args.model_path,
    base_dir = args.base_dir,
    aug=1,
    lps=0)

if args.dataset_name == "InHARD_13":
    config.warmup_steps = 5
    config.fixed_epoch = 5
    config.chunk_size = 2 # window for feature augmentation
    config.max_frames_per_video = 7360 #26342 #(inhard-4) #19330, (inhard-3) #12976,   #26342 (inhard-13) # 7361 
    config.learning_rate = 0.001 #1e-4 
    config.weight_decay = 3e-3
    config.batch_size = 5
    config.num_class = 14 #{3, 4, 14}
    config.d_model = 21
    #config.back_gd = ['']
    config.back_gd = ['No action']
    config.ensem_weights = [1, 1, 1, 1, 0]
elif args.dataset_name == "IKEA-ASM":
    config.d_model = 25
    config.chunk_size = 2 # window for feature augmentation
    config.max_frames_per_video = 993
    config.learning_rate = 0.001 #1e-4
    config.weight_decay = 3e-3
    config.batch_size = 32
    config.num_class = 14 #33
    #config.back_gd = ['']
    config.back_gd = ['NA']
    config.ensem_weights = [1, 1, 1, 1, 0]

config.output_dir = config.base_dir + "results/supervised_C2FTCN/"
if not os.path.exists(config.output_dir):
    os.mkdir(config.output_dir)

config.output_dir = config.output_dir + "split{}".format(config.split_number)

if args.wd is not None:
    config.weight_decay = args.wd
    config.output_dir=config.output_dir + "_wd{:.5f}".format(config.weight_decay)

if args.lr is not None:
    config.learning_rate = args.lr
    config.output_dir=config.output_dir + "_lr{:.6f}".format(config.learning_rate)

if args.chunk_size is not None:
    config.chunk_size = args.chunk_size
    config.output_dir=config.output_dir + "_chunk{}".format(config.chunk_size)

if args.ensem_weights is not None:
    config.output_dir=config.output_dir + "_wts{}".format(args.ensem_weights.replace(',','-'))
    config.ensem_weights = list(map(int, args.ensem_weights.split(",")))
    print("C2F Ensemble Weights being used is ", config.ensem_weights)


print("printing in output dir = ", config.output_dir)
config.project_name="{}-split{}".format(config.dataset, config.split_number)
config.train_split_file = config.base_dir + "splits/train.split{}.bundle".format(config.split_number)
config.test_split_file = config.base_dir + "splits/validation.split{}.bundle".format(config.split_number)
config.features_file_name = config.base_dir + "/features/inhard-13/30fps_p_light/"

if args.ft_file is not None:
    config.features_file_name = os.path.join(config.base_dir, args.ft_file)
    config.output_dir = config.output_dir + "_ft_file{}".format(args.ft_file)

if args.ft_size is not None:
    config.feature_size = args.ft_size
    config.output_dir = config.output_dir + "_ft_size{}".format(args.ft_file)
  
config.ground_truth_files_dir = config.base_dir + "/groundTruth/bvh_30fps/" 
config.label_id_csv = config.base_dir + 'mapping.csv'

config.output_dir = config.output_dir + "/"

if args.out_dir is not None:
    config.output_dir = args.out_dir + "/"

def model_pipeline(config):
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer, scheduler, postprocessor = make(config)

    #print (model)
    # and use them to train the model
    train(model, train_loader, criterion, optimizer, scheduler, config, test_loader, postprocessor)

    # and test its final performance
    model.load_state_dict(load_avgbest_model(config))
    acc = test(model, test_loader, criterion, postprocessor, config, config.epochs, 'avg')

    model.load_state_dict(load_best_model(config))
    acc = test(model, test_loader, criterion, postprocessor, config, config.epochs, '')

    return model

def load_best_model(config):
    return torch.load(config.output_dir + '/best_' + config.dataset + '_unet.wt')

def load_avgbest_model(config):
    return torch.load(config.output_dir + '/avgbest_' + config.dataset + '_unet.wt')

class CustomSchedule(_LRScheduler):
    def __init__(self, optimizer, d_model=21, warmup_steps=5, last_epoch=-1):
        self.d_model = d_model
        self.d_model = torch.tensor(self.d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        #print('inside lr funct')
        #print('self.base_lrs: ',  self.base_lrs)
        step = max(1, self.last_epoch + 1)  # avoid zero step
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        scale = self.d_model ** -0.5 * min(arg1, arg2)

         # Ensure `scale` is a float
        if isinstance(scale, torch.Tensor):
            scale = scale.item()

        # Calculate the new learning rates as a list of scalars
        new_lr = [base_lr * scale for base_lr in self.base_lrs]

        # Convert tensors in `new_lr` to floats
        new_lr_values = [lr.item() if isinstance(lr, torch.Tensor) else lr for lr in new_lr]
        #print('new_lr_values: ', new_lr_values)

        return new_lr_values
    
def make(config):
    # Make the data
    train, test = get_data(config, train=True), get_data(config, train=False)
    train_loader = make_loader(train, batch_size=config.batch_size, train=True)
    test_loader = make_loader(test, batch_size=config.batch_size, train=False)

    # Make the model
    model = get_model(config).to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(config)
    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                 betas=(0.9, 0.98), eps=1e-9
                                 )
    
    # Assuming d_model is already defined
    custom_scheduler = CustomSchedule(optimizer=optimizer, d_model=config.d_model)
    
    # postprocessor declaration
    postprocessor = PostProcess(config)
    postprocessor = postprocessor.to(device)
    
    return model, train_loader, test_loader, criterion, optimizer, custom_scheduler, postprocessor

class CriterionClass(nn.Module):
    def __init__(self, config):
        super().__init__()
        #Class-specific weights (you can adjust these according to your requirements)
        class_weights = torch.tensor([1.0, 20.0, 7.0, 6.0, 7.0, 5.0, 3.0, 39.0, 44.0, 8.0, 25.0, 1.0, 47.0, 13.0]) # 14 actions
        #class_weights = torch.tensor([1.0, 5.0, 4.0, 1.0]) # inhard-4
        #class_weights = torch.tensor([5.0, 2.0, 1.0]) # inhard-3

        #IKEA ASM 33 actions HCS
        #class_weights = torch.tensor([3.0, 11.0, 29.0, 19.0, 26.0, 6.0, 410.0, 10.0, 46.0, 38.0, 1662.0, 321.0, 889.0, 664.0, 409.0, 2545.0, 848.0, 35.0, 107.0, 85.0, 270.0, 10.0, 86.0, 24.0, 46.0, 402.0, 84.0, 723.0, 1204.0, 33.0, 25.0, 1.0, 14.0])
        #class_weights = torch.tensor([40.0, 51.0, 466.0, 38.0, 7338.0, 35.0, 1030.0, 84.0, 1927.0, 33.0, 25.0, 1.0, 14.0, 3.0]) #14 verbs


        '''class_weights = torch.tensor([0.0205118, 0.4130183, 0.148845, 0.1303033, 0.1341922, 0.1060895, 
                                      0.0702484, 0.7961783, 0.9090909, 0.1573812, 0.5062778, 0.0200329, 
                                      0.9619084, 0.2701826])'''

        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights.to(device))  # Frame wise cross entropy loss
        #self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')           # Migitating transistion loss 
    
    def forward(self, outp, labels, src_mask, labels_present):

        outp_wo_softmax = torch.log(outp + 1e-10).to(device)       # log is necessary because ensemble gives softmax output
        labels = labels.to(device)
        print(device)


        ce_loss = self.ce(outp_wo_softmax, labels)      
        
        mse_loss = 0.15 * torch.mean(torch.clamp(self.mse(outp_wo_softmax[:, :, 1:],
                                                          outp_wo_softmax.detach()[:, :, :-1]), 
                                                 min=0, max=16) * src_mask[:, :, 1:])

        loss = ce_loss + mse_loss 
        return {'full_loss':loss, 'ce_loss':ce_loss, 'mse_loss': mse_loss} 

def get_criterion(config):
    return CriterionClass(config)

def get_data(args, train=True):
    if train is True:
        fold='train'
        split_file_name = args.train_split_file
    else:
        fold='val'
        split_file_name = args.test_split_file

    dataset = AugmentDataset(args, fold=fold, fold_file_name=split_file_name, zoom_crop=(0.5, 2))
    return dataset

def make_loader(dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)
    return loader

def get_model(config):
    my_module = importlib.import_module(config.model_path)
    set_seed()
    return my_module.C2F_TCN(config.feature_size, config.num_class)

def get_c2f_ensemble_output(outp, weights):
    
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)
    #print('ensemble_prob: ', ensemble_prob.shape)

    for i, outp_ele in enumerate(outp[1]):
        #print('outp_ele: ', outp_ele.shape) 
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        #print('uppsampled outp_ele: ', upped_logit.shape)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)
        #print('ensemble_prob: ', ensemble_prob.shape)

    return ensemble_prob

def train(model, loader, criterion, optimizer, scheduler, config, test_loader, postprocessor):

    best_acc = 0
    avg_best_acc = 0
    accs = []
    no_improvement_epochs = 0  # Counter for epochs without improvement
    
    for epoch in range(config.epochs):
        start = time.time()
        model.train()
        #print('loader: ', len(loader))
        for i, item in enumerate(loader):
            #print('batch: ',i+1)
            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            #print('count: ', count)
            labels = item[2].to(device)
            src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
            src_mask = src_mask.to(device)
            
            src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

            # Forward pass ➡
            #print('samples: ',samples.shape)
            outputs_list = model(samples)

            outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)


            try:
                # Ensure outputs and labels are finite
                if not torch.isfinite(outputs_ensemble).all():
                    raise ValueError("Non-finite values found in outputs_ensemble")
                if not torch.isfinite(labels).all():
                    raise ValueError("Non-finite values found in labels")
    
                # Calculate loss
                loss_dict = criterion(outputs_ensemble, labels, src_msk_send, item[6].to(device))
                loss = loss_dict['full_loss']
    
                # Backward pass ⬅
                optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    loss.backward()

                # Step with optimizer
                optimizer.step()

            except Exception as e:
                print(f"Error during training: {e}")

            if i % 10 == 0:
                end = time.time()
                print(f"Train loss after {epoch} epochs, time: {end - start}, {i} iterations is {loss_dict['full_loss']:.3f}")
            
        # Update the scheduler
        scheduler.step()       
            

        acc, avg_score = test(model, test_loader, criterion, postprocessor, config, epoch, '')
        if avg_score > avg_best_acc:
            avg_best_acc = avg_score
            torch.save(model.state_dict(), config.output_dir + '/avgbest_' + config.dataset + '_unet.wt')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.output_dir + '/best_' + config.dataset + '_unet.wt')
            no_improvement_epochs = 0  # Reset counter if there is an improvement
        else:
            no_improvement_epochs += 1  # Increment counter if no improvement'''

        torch.save(model.state_dict(), config.output_dir + '/last_' + config.dataset + '_unet.wt')
        accs.append(acc)
        accs.sort(reverse=True)

        
        print(f'Validation best accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')
        print('-------------------------------------------------')

        
        '''# If warmup is done, switch to the post-warmup scheduler
        if epoch >= config.warmup_epochs:
            post_warmup_scheduler.step()'''

        
        # Check for early stopping
        if no_improvement_epochs >= 100:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in accuracy for 100 epochs.")
            break

def test(model, test_loader, criterion, postprocessors, args, epoch, dump_prefix):
    model.eval()
    print('in test')

    # Run the model on some test examples
    with torch.no_grad():
        start = time.time()
        correct, total = 0, 0
        avg_loss = []
        for i, item in enumerate(test_loader):
            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            labels = item[2].to(device)
            src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
            src_mask = src_mask.to(device)
            
            src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

            # Forward pass ➡
            outputs_list = model(samples)
            outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)
            
            loss_dict = criterion(outputs_ensemble, labels, src_msk_send, item[6].to(device))
            loss = loss_dict['full_loss']
            avg_loss.append(loss.item())
            
            pred = torch.argmax(outputs_ensemble, dim=1)
            #print('preds: ', pred.shape)

            correct += float(torch.sum((pred == labels) * src_mask).item())
            total += float(torch.sum(src_mask).item())
            postprocessors(outputs_ensemble, item[5], labels, count)
            
        # Add postprocessing and check the outcomes
        path = os.path.join(args.output_dir, dump_prefix + "predict_" + args.dataset)
        if not os.path.exists(path):
            os.mkdir(path)
        postprocessors.dump_to_directory(path)
        final_edit_score, map_v, overlap_scores = calculate_mof(args.ground_truth_files_dir, path, config.back_gd)
        postprocessors.start()
        acc = 100.0 * correct / total
        end = time.time()
        val_loss = np.mean(np.array(avg_loss))
        print(f"Validation loss = {val_loss: .3f}, accuracy of the model after epoch {epoch}, time {end - start} = {acc: .3f}%")
        
        with open(config.output_dir + "/results_file.txt", "a+") as fp:
            fp.write("{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n".format(overlap_scores[0], overlap_scores[1], 
                                                overlap_scores[2], final_edit_score, map_v))
        if epoch == config.epochs:
            with open(config.output_dir + "/" + dump_prefix + "final_results_file.txt", "a+") as fp:
                fp.write("{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n".format(overlap_scores[0], overlap_scores[1], 
                                                    overlap_scores[2], final_edit_score, map_v))
                

    avg_score = (map_v + final_edit_score) / 2
    #avg_score = (avg_score + overlap_scores[1] + 2*overlap_scores[2]) /4 # added
    return map_v, avg_score

import time

start_time = time.time()
model = model_pipeline(config)

end_time = time.time()

duration = (end_time - start_time) / 60
print(f"Total time taken = ", duration, "mins")

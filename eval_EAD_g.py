import importlib
import os
import time
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
from testtime_postprocess import PostProcess
import torch.nn.functional as F
from testtime_dataset_inHard_13 import AugmentDataset, collate_fn_override



my_parser = argparse.ArgumentParser()
my_parser.add_argument('--dataset_name', type=str, default="breakfast", choices=['InHARD_13', 'IKEA-ASM' ])
my_parser.add_argument('--split', type=int, required=False, help="Comma seperated split number to run evaluation," + \
                                                                  "default = 1,2,3,4 for breakfast and gtea, 1,2,3,4,5 for 50salads")
my_parser.add_argument('--cudad', type=str, default='0', help="Cuda device number to run evaluation program in")
my_parser.add_argument('--base_dir', type=str, help="Base directory containing groundTruth, features, splits directory of dataset")
my_parser.add_argument('--chunk_size', type=int, required=False, help="Provide chunk size which as used for training," + \
                                                                      "by default it is set for datase")
my_parser.add_argument('--ensem_weights', type=str, required=False,
                        help='Default = \"1,1,1,1,0,0\", provide in similar format comma-seperated 6 weights values if required to be changed')
my_parser.add_argument('--ft_file', type=str, required=False, help="Provide feature file dir path if default is not base_dir/features")
my_parser.add_argument('--ft_size', type=int, required=False, help="Default = 2048 for the I3D features, change if feature size changes")
my_parser.add_argument('--model_path', type=str, default='model_5_ed_infoGcn_b_v2')
my_parser.add_argument('--err_bar', type=int, required=False)
my_parser.add_argument('--compile_result', action='store_true', help="To get results without test time augmentation use --compile_result")
my_parser.add_argument('--num_workers', type=int, default=0, help="Number of workers to be used for data loading")
my_parser.add_argument('--out_dir', required=False, help="Directory where output(checkpoints, logs, results) is to be dumped")
my_parser.add_argument('--model_checkpoint', required=False, help="Checkpoint to pick up the model")
args = my_parser.parse_args()


seed = 42

if args.err_bar:
    seed = args.err_bar #np.random.randint(0, 999999)

if args.model_checkpoint:
    split_dir = args.model_checkpoint.split("/")
    args.out_dir = "/".join(split_dir[:-2])
    print(f"With model checkpoint {args.model_checkpoint}, output directory is {args.out_dir}")

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


config = dotdict(
    epochs=500,
    dataset=args.dataset_name,
    feature_size = 63, #inhard_13 orientation, pose light
    gamma=0.5,
    step_size=500,
    model_path=args.model_path,
    base_dir =args.base_dir,
    aug=1,
    lps=0)


config.ensem_weights = [1, 1, 1, 1, 0]

if  args.dataset_name == "InHARD_13":
    config.chunk_size = 2
    config.max_frames_per_video = 7360 #26342 #19330 #(inhard-4) #13005 #(inhard-3)
    config.learning_rate = 1e-4
    config.weight_decay = 3e-3
    config.batch_size = 3
    config.num_class = 14
    #config.back_gd = ['']
    config.back_gd = ['No action']
    config.split = [1]
    if not args.compile_result:
        config.chunk_size = list(range(1, 10))
        config.weights = np.ones(len(config.chunk_size))
    else:
        config.chunk_size = [2]
        config.weights = [1]
        config.eval_true = True

elif args.dataset_name == "IKEA-ASM":
    config.chunk_size = 2
    config.max_frames_per_video = 993 #19330 #13005
    config.learning_rate = 0.001
    config.weight_decay = 3e-3
    config.batch_size = 32
    config.num_class = 33
    config.back_gd = ['NA']
    config.split = [1]
    if not args.compile_result:
        config.chunk_size = list(range(1, 10))
        config.weights = np.ones(len(config.chunk_size))
    else:
        config.chunk_size = [2]
        config.weights = [1]
        config.eval_true = True

    
if args.split is not None:
    try:
        args.split = int(args.split)
        config.split = [args.split]
    except:
        config.split = args.split.split(',')

config.features_file_name = config.base_dir + "/features/inhard-13/30fps_p_light" #"/features/inhard-13/30fps_p/"
config.ground_truth_files_dir = config.base_dir + "/groundTruth/bvh_30fps/"
config.label_id_csv = config.base_dir + 'mapping.csv'


def model_pipeline(config):
    acc_list = []
    edit_list = []
    f1_10_list = []
    f1_25_list = []
    f1_50_list = []
    for ele in config.split:
        config.output_dir = config.base_dir + "results/supervised_C2FTCN/split{}".format(ele) #, config.model_path, ele, config.aug)
        # if args.wd is not None:
        #     config.weight_decay = args.wd
        #     config.output_dir=config.output_dir + "_wd{:.5f}".format(config.weight_decay)

        # if args.lr is not None:
        #     config.learning_rate = args.lr
        #     config.output_dir=config.output_dir + "_lr{:.6f}".format(config.learning_rate)

        if args.chunk_size is not None:
            config.chunk_size = args.chunk_size
            config.output_dir = config.output_dir + "_chunk{}".format(config.chunk_size)

        if args.ensem_weights is not None:
            config.output_dir = config.output_dir + "_wts{}".format(args.ensem_weights.replace(',', '-'))
            config.ensem_weights = list(map(int, args.ensem_weights.split(",")))
            print("Weights being used is ", config.ensem_weights)

        config.output_dir = config.output_dir + "/"
        if args.out_dir is not None:
            config.output_dir = args.out_dir + "/"

        print("printing getting the output from output dir = ", config.output_dir)
        config.project_name="{}-split{}".format(config.dataset, ele)
        config.test_split_file = config.base_dir + "splits/test.split{}.bundle".format(ele)
        # make the model, data, and optimization problemè()
        model, test_loader, postprocessor = make(config)
        print('am here !---------------')
        model.load_state_dict(load_best_model(config))
        prefix = ''

        # model.eval()

        start = time.time()

        correct, correct1, total = 0, 0, 0
        postprocessor.start()

        with torch.no_grad():
            for i, item in enumerate(test_loader):
                samples = item[0].to(device).permute(0,2,1)
                count = item[1].to(device)
                labels = item[2].to(device)
                src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                src_mask = src_mask.to(device)

                outplist = model(samples)
                ensembel_out = get_ensemble_out(outplist)

                pred = torch.argmax(ensembel_out, dim=1)
                correct += float(torch.sum((pred==labels)*src_mask).item())
                total += float(torch.sum(src_mask).item())

                # postprocessor(ensembel_out, item[5], labels, count)
                # 7 chunk size, 8 is chunk id
                postprocessor(ensembel_out, item[5], labels, count, item[7].to(device), item[8], item[3].to(device)) 

        print(f'Accuracy: {100.0*correct/total: .2f}')
         # Add postprocessing and check the outcomes
        path = os.path.join(config.output_dir, prefix + "testtime_augmentation_split{}".format(ele))
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Output files will be dumped in {path} directory")
        postprocessor.dump_to_directory(path)
    
        final_edit_score, map_v, overlap_scores = calculate_mof(config.ground_truth_files_dir, path, config.back_gd)
        acc_list.append(map_v)
        edit_list.append(final_edit_score)
        f1_10_list.append(overlap_scores[0])
        f1_25_list.append(overlap_scores[1])
        f1_50_list.append(overlap_scores[2])
        
        end = time.time()
        print(" Done! time: {:0.2f} sec".format(end - start))
    
    

    print("Frame accuracy = ", np.mean(np.array(acc_list)))
    print("Edit Scores = ", np.mean(np.array(edit_list)))
    print("f1@10 Scores = ", np.mean(np.array(f1_10_list)))
    print("f1@25 Scores = ", np.mean(np.array(f1_25_list)))
    print("f1@50 Scores = ", np.mean(np.array(f1_50_list)))


def load_best_model(config):
    if args.model_checkpoint is not None:
        print(f"Loading checkpoint from {args.model_checkpoint}")
        return torch.load(args.model_checkpoint)
    checkpoint_file = config.output_dir + '/best_' + config.dataset + '_unet.wt'
    print(f"Loading checkpoint from {checkpoint_file}")
    return torch.load(checkpoint_file)

def load_avgbest_model(config):
    if args.model_checkpoint is not None:
        return torch.load(args.model_checkpoint)
    return torch.load(config.output_dir + '/avgbest_' + config.dataset + '_unet.wt')

def make(config):
    # Make the data
    test = get_data(config, train=False)
    test_loader = make_loader(test, batch_size=config.batch_size, train=False)

    # Make the model
    model = get_model(config).to(device)
    #print(model)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # postprocessor declaration
    postprocessor = PostProcess(config, config.weights)
    postprocessor = postprocessor.to(device)
    
    return model, test_loader, postprocessor


def get_data(args, train=True):
    if train is True:
        fold='train'
        split_file_name = args.train_split_file
    else:
        fold='val'
        split_file_name = args.test_split_file
        print('split_file_name: ', split_file_name)
    dataset = AugmentDataset(args, fold=fold, fold_file_name=split_file_name, chunk_size=config.chunk_size)
    
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


def get_ensemble_out(outp):
    
    weights = config.ensem_weights
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

    for i, outp_ele in enumerate(outp[1]):
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)
    
    return ensemble_prob

model = model_pipeline(config)

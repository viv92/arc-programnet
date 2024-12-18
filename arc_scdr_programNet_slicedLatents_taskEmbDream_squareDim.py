import os
import cv2
import math 
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np
import torch
torch.set_float32_matmul_precision('high') # use TF32 precision for speeding up matmul
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from time import time 

from utils_arc_scdr_programNet_slicedLatents_taskEmbDream_squareDim import *


# function to prepare dataset 
def prepare_arc_data_square(chal_file, sol_file, n, num_examples):
    dataset = []

    # load eval chal and sol jsons as dicts
    with open(chal_file) as f:
        eval_chal_dict = json.load(f)
    with open(sol_file) as f:
        eval_sol_dict = json.load(f)

    # prepare tune dataset 
    for qid in eval_chal_dict.keys():

        # create tune dataset items
        examples = []
        for i in range(len(eval_chal_dict[qid]['train'])):
            x = eval_chal_dict[qid]['train'][i]['input']
            y = eval_chal_dict[qid]['train'][i]['output']
            x_shape = np.array(x).shape 
            y_shape = np.array(y).shape 
            if x_shape[0] == n and x_shape[1] == n and y_shape[0] == n and y_shape[1] == n:
                x = torch.tensor(x).flatten()
                y = torch.tensor(y).flatten()
                example = torch.cat([x, y], dim=-1)
                examples.append(example)

        # ensure same number of examples per task (for parallelism)
        if len(examples) > 0:
            examples = examples[:num_examples]
            while len(examples) < num_examples:
                examples.append(example)

        # create test dataset items
        queries, answers = [], []
        for i in range(len(eval_chal_dict[qid]['test'])):
            x = eval_chal_dict[qid]['test'][i]['input']
            y = eval_sol_dict[qid][i]
            x_shape = np.array(x).shape 
            y_shape = np.array(y).shape 
            if x_shape[0] == n and x_shape[1] == n and y_shape[0] == n and y_shape[1] == n:
                x = torch.tensor(x).flatten()
                y = torch.tensor(y).flatten()
                queries.append(x)
                answers.append(y)

        L = len(examples) * len(queries) 
        if L > 0: # valid square task

            # convert to tensors 
            examples = torch.stack(examples).long()
            queries = torch.stack(queries).long()
            answers = torch.stack(answers).long()

            # prepare item dict
            for i in range(queries.shape[0]): # one item per query
                data_item = {}
                data_item['examples'] = examples 
                data_item['query'] = queries[i] 
                data_item['answer'] = answers[i] 
                data_item['qid'] = qid 
                dataset.append(data_item) 

    return dataset # list of dicts


def reconstruction_loss(recon_x, x):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    recon_x = recon_x.permute(0,2,1) # [b, vocab, seqlen]
    recon_loss = criterion(recon_x, x)
    return recon_loss

def compression_loss(z):
    criterion = nn.MSELoss(reduction='mean')
    # latent compress loss - drive latents to zero (pad) latent
    compress_targets = torch.zeros_like(z)
    compress_loss = criterion(z, compress_targets)
    return compress_loss

def task_reconstruction_loss(recon_x, x):
    criterion = nn.MSELoss(reduction='mean')
    recon_loss = criterion(recon_x, x)
    return recon_loss

def ema(arr, val, r=0.01):
    if len(arr) == 0:
        return [val]
    newval = arr[-1] * (1-r) + val * r 
    arr.append(newval)
    return arr 


# function to visualize predicted grids 
def visualize_grids(dataset, savepath, square_dim):

    for item in dataset:
        x = item['query']
        y = item['answer']
        pred_y = item['pred_answer']
        qid = item['qid']

        x = x.squeeze().view(square_dim, square_dim).cpu().numpy() 
        y = y.squeeze().view(square_dim, square_dim).cpu().numpy() 
        pred_y = pred_y.squeeze().view(square_dim, square_dim).cpu().numpy()  

        # Define the color map
        colors = ['#000000', '#0000FF', '#FF0000', '#00FF00', '#FFFF00', '#808080', '#FF00FF', '#FFA500', '#00FFFF', '#800000', '#FFFFFF']  # Black, Blue, Red, Green, Yellow, Gray, Magenta, Orange, Cyan, Maroon, White
        cmap = mcolors.ListedColormap(colors)

        # Create a new figure with four subplots in a 2x2 grid
        fig, axs = plt.subplots(1, 3, figsize=(16, 8))

        # Function to plot a single grid
        def plot_grid(ax, matrix, title, square_dim):
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(which='major', axis='both', linestyle='-', color='black', linewidth=0.5)
            ax.set_xticks(np.arange(-.5, square_dim-1, 1), minor=True)
            ax.set_yticks(np.arange(-.5, square_dim-1, 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(title)

        # Plot all four grids
        plot_grid(axs[0], x, "x", square_dim)
        plot_grid(axs[1], y, "y", square_dim)
        plot_grid(axs[2], pred_y, "pred_y", square_dim)

        # if x2_test is not None:
        #     plot_grid(axs[1, 0], x2_test, "x2_test", square_dim)
        #     plot_grid(axs[1, 1], y2_test, "y2_test", square_dim)
        #     plot_grid(axs[1, 2], pred_y2_test, "pred_y2_test", square_dim)

        # Adjust the layout and display
        # plt.tight_layout()
        fig.suptitle('qid: ' + qid)
        fig.savefig(savepath + '_qid=' + qid + '.png')
        plt.close(fig)



def discretize_weights(weights, interval, lower_limit):
    weights = (weights - lower_limit) / interval
    return weights.long()

def undiscretize_weights(weights, interval, lower_limit):
    weights = weights * interval + lower_limit
    return weights.float()
    
def set_weights(model, weights, interval, lower_limit):
    weights = weights.flatten()

    # undiscretize weights
    weights = undiscretize_weights(weights, interval, lower_limit)

    current_index = 0
    
    for name, param in model.named_parameters():
        if (name != 'emb.weight') and (param.requires_grad):  # If the parameter is trainable
            # Get the number of elements in the parameter
            num_params = param.numel()
            
            # Extract the corresponding slice from the flat tensor
            new_values = weights[current_index:current_index + num_params]
            
            # Reshape the new_values to match the parameter's shape
            new_values = new_values.view_as(param)
            
            # Assign the new values to the parameter's data
            param.data = new_values
            
            # Update the index to the next set of values
            current_index += num_params

    # fixed emb weights - to xavier initialized

    # emb_weights = torch.linspace(0.01, pn_upper_limit - 0.01, arc_vocab_size, device=model.device).float()
    # emb_weights = emb_weights.repeat(1, pn_d_model)
    # model.emb.weight.data = emb_weights



## main 
if __name__ == '__main__':

    # hyperparams for quantization
    num_quantized_values = [7, 5, 5, 5] # L in fsq paper (use odd values only)
    latent_dim = len(num_quantized_values)

    # hyperparams for vocab and bottleneck dims 
    square_dim = 3 # 10
    arc_vocab_size = 10
    dit_vocab_size = 1
    for n in num_quantized_values:
        dit_vocab_size *= n # equal to latent codebook size
    dit_vocab_size += 1 # for mask token
    dit_mask_token = dit_vocab_size - 1
    num_examples = 4 # fixed number of examples for each task - required for parallelism 
    one_example_emb_dim = 16 # 64
    one_example_emb_seqlen = square_dim * square_dim #* 2
    example_emb_dim = 16 # 64 
    example_emb_seqlen = 4 # one_example_emb_seqlen * num_examples

    # hyperparams for latent slicing 
    program_latent_seqlen = 32 # 64

    # hyperparams for all Transformers
    d_model = 128
    n_heads = 8
    assert d_model % n_heads == 0
    d_k = d_model // n_heads 
    d_v = d_k 
    n_layers = 3 # 6
    d_ff = d_model * 4

    # hyperparams for dit 
    dit_d_model = 512 # 128 
    dit_n_heads = 8
    assert dit_d_model % dit_n_heads == 0
    dit_d_k = dit_d_model // dit_n_heads
    dit_d_v = dit_d_k 
    dit_n_layers = 6 # 3
    dit_d_ff = dit_d_model * 4

    # hyperparams for task embedder 
    te_d_model = 16 # 64
    te_n_heads = 4
    assert te_d_model % te_n_heads == 0
    te_d_k = te_d_model // te_n_heads
    te_d_v = te_d_k 
    te_n_layers = 2 # 3
    te_d_ff = te_d_model * 4

    # hyperparams for program net 
    pn_d_model = 8 # 2 # 16
    pn_n_heads = 1 # 4 # 2
    assert pn_d_model % pn_n_heads == 0
    pn_d_k = pn_d_model // pn_n_heads
    pn_d_v = pn_d_k 
    pn_n_layers = 2 # 1

    # hyperparams for program net weights discretization
    pn_upper_limit = 1
    pn_lower_limit = -1 
    pn_num_levels = 32 # 512 # 128
    pn_interval = (pn_upper_limit - pn_lower_limit) / pn_num_levels

    dropout = 0.1 
    weight_decay = 0.1 
    compress_factor = 1 # 0 # 0.01 # 0.1
    reconstruction_factor = 1
    prediction_factor = 0.1 

    # hyperparams for sleep mode
    sleep_mode = 0 # 0 = wake, 1 = sleep, 2 = dream    
    sleep_steps = 1000 # 16000 
    wake_steps = 300
    dream_steps = 1000 # 64000  
    num_switches = -1
    sleep_steps_list = [wake_steps, sleep_steps, dream_steps]

    # hyperparams for training 
    diffusion_start_time_eps = 1e-3
    batch_size = 64
    gradient_accumulation_steps = 1 
    lr = 3e-4
    num_cycles = 20000
    num_train_steps = sum(sleep_steps_list) * num_cycles
    train_steps_done = 0
    random_seed = 10
    # resume_training_from_ckpt = False   

    # hyperparams for figures and plotting
    sampling_freq = (720 * 1) - 1 
    plot_freq = sampling_freq * 4

    # hyperparams for fantasies
    fantasy_batch_size = 32 
    fantasy_tries = 16 # 4
    max_fantasies = fantasy_batch_size * fantasy_tries # number of fantasies added in one infusion
    all_fantasy_dataset_size = 10 ** 6
    delay_start_iter = 100
    fantasy_gen_cycle_counter = -1
    fantasy_gen_skip_cycles = 1 

    sampling_eps = 0.1
    sleep_loss_threshold = 0
    dream_loss_threshold = 0
    timeout = 60 * 5

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init program net 
    program_net = init_program_net(square_dim * square_dim, pn_d_model, pn_n_heads, pn_d_k, pn_d_v, pn_n_layers, arc_vocab_size, device).to(device)
    program_net.eval()

    # calculate program seq len = number of params in program net
    program_seq_len = sum(p.numel() for p in program_net.parameters() if p.requires_grad)

    # fixed embedding case 
    program_seq_len -= arc_vocab_size * pn_d_model
    print('program_net.emb.weight.data: ', program_net.emb.weight.data)

    # hyperparams for sampling
    num_sampling_steps = int(program_latent_seqlen * 0.25)
    wake_batch_size = 32
    p_uncond = 0.1 
    cfg_scale = 2.0

    # create hyperparam str
    hyperparam_dict = {}
    hyperparam_dict['method'] = 'programNetFixedEmb_lossSwInc_timeoutInit_smallTE_fant2randInit_TED_sampEps0.1' 
    hyperparam_dict['sqD'] = square_dim
    hyperparam_dict['fsqD'] = d_model
    hyperparam_dict['fsqL'] = n_layers
    hyperparam_dict['pD'] = pn_d_model
    hyperparam_dict['pH'] = pn_n_heads
    hyperparam_dict['pL'] = pn_n_layers 
    hyperparam_dict['pSeq'] = program_seq_len
    hyperparam_dict['pLatSeq'] = program_latent_seqlen
    hyperparam_dict['pLim'] = pn_upper_limit 
    hyperparam_dict['pLev'] = pn_num_levels  
    hyperparam_dict['B'] = batch_size 
    # hyperparam_dict['lr'] = lr
    # hyperparam_dict['ex1D'] = one_example_emb_dim
    # hyperparam_dict['ex1L'] = one_example_emb_seqlen
    hyperparam_dict['exD'] = example_emb_dim
    hyperparam_dict['exL'] = example_emb_seqlen
    # hyperparam_dict['Wdecay'] = weight_decay
    hyperparam_dict['drop'] = dropout
    hyperparam_dict['CF'] = compress_factor
    # hyperparam_dict['RF'] = reconstruction_factor
    # hyperparam_dict['PF'] = prediction_factor
    # hyperparam_dict['initMode'] = sleep_mode
    # hyperparam_dict['sleep'] = sleep_steps
    # hyperparam_dict['dream'] = dream_steps
    hyperparam_dict['wake'] = wake_steps
    # hyperparam_dict['sampSteps'] = num_sampling_steps
    # hyperparam_dict['delay'] = delay_start_iter 
    hyperparam_dict['Fnum'] = max_fantasies 
    hyperparam_dict['Fskips'] = fantasy_gen_skip_cycles 
    hyperparam_dict['Fbuf'] = all_fantasy_dataset_size 
    # hyperparam_dict['swS'] = sleep_loss_threshold 
    # hyperparam_dict['swD'] = dream_loss_threshold

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + '=' + str(v)


    results_dir = './results/' + hyperparam_str + '/'
    ckpts_dir = './ckpts/'
    # dit_ckpt_path = ckpts_dir + 'dit_' + hyperparam_str + '.pt'
    # fsq_ckpt_path = ckpts_dir + 'fsq_' + hyperparam_str + '.pt'
    # task_embedder_ckpt_path = ckpts_dir + 'task_embedder_' + hyperparam_str + '.pt'
    # task_inverter_ckpt_path = ckpts_dir + 'task_inverter_' + hyperparam_str + '.pt'
      
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # load data
    folder = '/home/vivswan/experiments/arc_needs_system2/arc-prize-2024/'
    train_chal_file = folder + 'arc-agi_training_challenges.json'
    train_sol_file = folder + 'arc-agi_training_solutions.json'
    train_dataset = prepare_arc_data_square(train_chal_file, train_sol_file, square_dim, num_examples)
    test_chal_file = folder + 'arc-agi_evaluation_challenges.json'
    test_sol_file = folder + 'arc-agi_evaluation_solutions.json'
    test_dataset = prepare_arc_data_square(test_chal_file, test_sol_file, square_dim, num_examples)

    # put all data in test 
    test_dataset.extend(train_dataset)
    train_dataset = []

    print('len(train_dataset): ', len(train_dataset))
    print('len(test_dataset): ', len(test_dataset))
    print('program_seq_len: ', program_seq_len)

    # init task embedder model
    example_encoder = init_example_encoder_transformer(arc_vocab_size, square_dim * square_dim * 2, te_d_model, te_d_k, te_d_v, te_n_heads, te_n_layers, te_d_ff, dropout, one_example_emb_dim, one_example_emb_seqlen, device)                                                 
    example_decoder = init_example_decoder_transformer(one_example_emb_dim, one_example_emb_seqlen * num_examples, te_d_model, te_d_k, te_d_v, te_n_heads, te_n_layers, te_d_ff, dropout, example_emb_dim, example_emb_seqlen, device)
    example_embedder = Example_Transformer(example_encoder, example_decoder, device).to(device)

    # init program FSQ 
    fsq_encoder = init_fsq_encoder_transformer(pn_num_levels, example_emb_dim, program_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, latent_dim, device)
    fsq_decoder = init_fsq_decoder_transformer(latent_dim, program_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, pn_num_levels, device)
    fsq = FSQ_Transformer(device, num_quantized_values, fsq_encoder, fsq_decoder, program_seq_len, program_latent_seqlen).to(device)

    # init dit 
    dit_x_seq_len = program_latent_seqlen
    dit_max_seq_len = dit_x_seq_len + 1 # [t, x]
    dit_condition_dim = example_emb_dim
    dit = init_dit(dit_max_seq_len, dit_x_seq_len, dit_d_model, dit_condition_dim, dit_vocab_size, dit_d_k, dit_d_v, dit_n_heads, dit_n_layers, dit_d_ff, dropout, device).to(device)

    # init optimizers
    sleep_params = fsq.parameters()
    sleep_optimizer = torch.optim.AdamW(sleep_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    
    dream_params = list(example_embedder.parameters()) + list(dit.parameters())
    dream_optimizer = torch.optim.AdamW(dream_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)


    # results for plotting
    results_test_items = [len(test_dataset)]
    results_dit_loss, results_dit_batch_accuracy = [], []
    results_wake_pixel_accuracy, results_wake_pair_accuracy = [], []
    results_train_loss, results_recon_loss, results_compress_loss = [], [], []
    results_codebook_usage, results_codebook_unique, results_masked_latents = [], [], []

    # train

    train_step = train_steps_done
    sleep_mode_counter = 0 
    pbar = tqdm(total=num_train_steps)

    solved_test_idx = [] # used to remove solved test items from test dataset
    new_fantasy_dataset = []
    all_fantasy_dataset = [] 
    wake_visualize_dataset = []
    
    while train_step < num_train_steps + train_steps_done:

        # handle sleep mode 
        if not (num_switches == 0):
            # switch_required = (sleep_mode_counter == sleep_steps_list[sleep_mode])
            switch_required = 0

            if (sleep_mode == 0) and (sleep_mode_counter == sleep_steps_list[sleep_mode]): # wake to sleep
                switch_required = 1
                start_time = time()

            if (sleep_mode == 1): # wake to sleep
                if sleep_loss_threshold == 0:
                    if time() - start_time > timeout:
                        switch_required = 1
                        sleep_loss_threshold = results_train_loss[-1] * 0.9
                        start_time = time()
                else:
                    if (sleep_mode_counter > sleep_steps_list[sleep_mode]) and (results_train_loss[-1] < sleep_loss_threshold): 
                        switch_required = 1
                        sleep_loss_threshold = results_train_loss[-1] * 0.9

            if (sleep_mode == 2): # sleep to dream
                if dream_loss_threshold == 0:
                    if time() - start_time > timeout:
                        switch_required = 1
                        dream_loss_threshold = results_dit_loss[-1] * 0.9
                else: 
                    if (sleep_mode_counter > sleep_steps_list[sleep_mode]) and (results_dit_loss[-1] < dream_loss_threshold): 
                        switch_required = 1
                        dream_loss_threshold = results_dit_loss[-1] * 0.9

            # if switch from wake to sleep, remove test items solved in wake phase
            if switch_required and (sleep_mode == 0):
                solved_test_idx = list(set(solved_test_idx)) # deduplicate
                test_dataset = [x for i,x in enumerate(test_dataset) if i not in solved_test_idx]
                solved_test_idx = []
                remaining_test_items = len(test_dataset)
                results_test_items.append(remaining_test_items)

            while switch_required:
                sleep_mode += 1
                sleep_mode = sleep_mode % len(sleep_steps_list)
                sleep_mode_counter = 0
                switch_required = (sleep_mode_counter == sleep_steps_list[sleep_mode]) 
                num_switches -= 1


        if (sleep_mode == 0) and (len(test_dataset) > 0): # wake mode - get solved test data 
            dit.eval()
            fsq.eval()
            example_embedder.eval()

            # fetch test minibatch
            test_idx = np.arange(len(test_dataset))
            np.random.shuffle(test_idx)
            test_idx = test_idx[:wake_batch_size]
            minibatch = [test_dataset[i] for i in test_idx]

            with torch.no_grad():

                # prepare target inputs and outputs (using [query, answer] pair as an extra example)
                target_inputs, target_outputs = [], []
                for item in minibatch:
                    example_inputs, example_outputs = item['examples'][:, :(square_dim**2)], item['examples'][:, -(square_dim**2):]
                    example_inputs = torch.cat([example_inputs, item['query'].unsqueeze(0)], dim=0) # [num_examples + 1, seqlen]
                    example_outputs = torch.cat([example_outputs, item['answer'].unsqueeze(0)], dim=0) # [num_examples + 1, seqlen]
                    target_inputs.append(example_inputs)
                    target_outputs.append(example_outputs)
                target_inputs, target_outputs = torch.stack(target_inputs).to(device), torch.stack(target_outputs).to(device) # [b, num_examples + 1, seqlen]

                # get example embeddings
                example_embs = example_embedder(minibatch) # [b, example_emb_seqlen, example_emb_dim]

                # get sample tokens corresponding to indices of codebook 
                x_sample = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, example_embs.shape[0], example_embs, cfg_scale, 0, device) # x_sample.shape: [b, seqlen]
                x_sample = x_sample.flatten() # shape: [b * seqlen]

                # get codebook vectors indexed by x_sample
                sampled_latents = fsq.codebook[x_sample] # sampled_latents.shape: [b * program_latent_seqlen, latent_dim]
                sampled_latents = sampled_latents.unflatten(dim=0, sizes=(-1, program_latent_seqlen)) # [b, program_latent_seqlen, latent_dim]
                padded_latents = torch.zeros(sampled_latents.shape[0], program_seq_len, sampled_latents.shape[-1], device=sampled_latents.device)
                padded_latents[:, :sampled_latents.shape[1]] = sampled_latents
                padded_latents = padded_latents.flatten(start_dim=0, end_dim=1) # [b * program_seq_len, latent_dim]
                pred_program_scores, _ = fsq.decode(padded_latents.float()) # [b, seqlen, num_levels]
                pred_program_weights = torch.argmax(pred_program_scores, dim=-1).long() # [b, seqlen]

                # forward prop through program net to get pred outputs 
                pred_outputs = []
                for b in range(pred_program_weights.shape[0]):
                    set_weights(program_net, pred_program_weights[b], pn_interval, pn_lower_limit)
                    outputs = program_net.predict(target_inputs[b]) # [num_examples + 1, seqlen]
                    pred_outputs.append(outputs)
                pred_outputs = torch.stack(pred_outputs).long().to(device) # [b, num_examples + 1, seqlen]

                # get solved minibatch idx
                bools = pred_outputs == target_outputs 
                pixel_accuracy = bools.float().mean()
                bools = bools.all(dim=-1) # [b, num_example + 1]
                pair_accuracy = bools.float().mean() 

                bools = bools.all(dim=-1) # [b]
                solved_minibatch_idx = torch.where(bools)[0]
                if len(solved_minibatch_idx) > 0:
                    print('solved_minibatch_idx: ', solved_minibatch_idx)

            # add solved test items to train dataset (including solving program weights)
            for i in solved_minibatch_idx:
                test_index = test_idx[i]
                test_item = test_dataset[test_index]
                test_item['program'] = pred_program_weights[i].cpu() # [seqlen]
                train_dataset.append(test_item)
                solved_test_idx.append(test_index) # to later remove the solved test items from test dataset

            # bookeeping for pixel and pair accuracy 
            results_wake_pair_accuracy.append(pair_accuracy.item())
            results_wake_pixel_accuracy.append(pixel_accuracy.item())

            # add to wake visualize dataset for visualization
            # for i in solved_minibatch_idx:
            y = target_outputs[0][-1]
            pred_y = pred_outputs[0][-1].cpu()
            x = minibatch[0]['query']
            qid = minibatch[0]['qid']
            vis_item = {}
            vis_item['query'] = x 
            vis_item['answer'] = y 
            vis_item['pred_answer'] = pred_y 
            vis_item['qid'] = qid 
            wake_visualize_dataset.append(vis_item)

            pbar.update(1)
            pbar.set_description('mode:{} test_items:{}'.format(sleep_mode, len(test_dataset)))
            dit.train()
            fsq.train()
            example_embedder.train()
    


        if sleep_mode == 1: # sleep mode - train FSQ 
            dit.eval()
            example_embedder.eval()

            ## generate fantasy dataset 
            if (sleep_mode_counter == 0) and (train_step > delay_start_iter):
                fantasy_gen_cycle_counter += 1

                # first ever fantasies should be random
                if fantasy_gen_cycle_counter == 0:

                    for j in range(fantasy_tries):

                        # fetch test minibatch
                        test_idx = np.arange(len(test_dataset))
                        np.random.shuffle(test_idx)
                        test_idx = test_idx[:fantasy_batch_size]
                        minibatch = [test_dataset[i] for i in test_idx]

                        with torch.no_grad():

                            # prepare target inputs (using only examples and not [query, answer] pair)
                            target_inputs = []
                            for item in minibatch:
                                example_inputs = item['examples'][:, :(square_dim**2)]
                                target_inputs.append(example_inputs)
                            target_inputs = torch.stack(target_inputs).to(device) # [b, num_examples, seqlen]

                            pred_program_weights = torch.randint(low=0, high=pn_num_levels, size=(len(minibatch), program_seq_len)).long().to(device) # [b, seqlen]

                            pred_outputs = []
                            for b in range(pred_program_weights.shape[0]):
                                set_weights(program_net, pred_program_weights[b], pn_interval, pn_lower_limit)
                                outputs = program_net.predict(target_inputs[b]) # [num_examples, seqlen]
                                pred_outputs.append(outputs)
                            pred_outputs = torch.stack(pred_outputs).long().to(device) # [b, num_examples, seqlen]

                            # prepare fantasy items and add to fantasy dataset
                            for i in range(len(minibatch)):
                                item = {}
                                item_examples = torch.cat([target_inputs[i], pred_outputs[i]], dim=-1) # [num_examples, seqlen * 2]
                                item_program = pred_program_weights[i] # [seqlen]
                                item['examples'] = item_examples.cpu()
                                item['program'] = item_program.cpu()
                                item['qid'] = 'fantasy'
                                new_fantasy_dataset.append(item)


                # not the first ever fantasies
                if (fantasy_gen_cycle_counter > 0) and (fantasy_gen_cycle_counter % fantasy_gen_skip_cycles == 0):
                    with torch.no_grad():

                        all_fantasy_dataset.extend(new_fantasy_dataset)
                        all_fantasy_dataset = all_fantasy_dataset[-all_fantasy_dataset_size:]
                        new_fantasy_dataset = [] # flush new fantasy dataset

                        for j in range(fantasy_tries):

                            # fetch test minibatch
                            test_idx = np.arange(len(test_dataset))
                            np.random.shuffle(test_idx)
                            test_idx = test_idx[:fantasy_batch_size]
                            minibatch = [test_dataset[i] for i in test_idx]

                            with torch.no_grad():

                                # prepare target inputs (using only examples and not [query, answer] pair)
                                target_inputs = []
                                for item in minibatch:
                                    example_inputs = item['examples'][:, :(square_dim**2)]
                                    target_inputs.append(example_inputs)
                                target_inputs = torch.stack(target_inputs).to(device) # [b, num_examples, seqlen]

                                # get example embeddings
                                example_embs = example_embedder(minibatch)

                                # get sample tokens corresponding to indices of codebook 
                                x_sample = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, example_embs.shape[0], example_embs, cfg_scale, sampling_eps, device) # x_sample.shape: [b, seqlen]
                                x_sample = x_sample.flatten() # shape: [b * seqlen]

                                # get codebook vectors indexed by x_sample
                                sampled_latents = fsq.codebook[x_sample] # sampled_latents.shape: [b, seqlen, latent_dim]
                                sampled_latents = sampled_latents.unflatten(dim=0, sizes=(-1, program_latent_seqlen)) # [b, program_latent_seqlen, latent_dim]
                                padded_latents = torch.zeros(sampled_latents.shape[0], program_seq_len, sampled_latents.shape[-1], device=sampled_latents.device)
                                padded_latents[:, :sampled_latents.shape[1]] = sampled_latents
                                padded_latents = padded_latents.flatten(start_dim=0, end_dim=1) # [b * program_seq_len, latent_dim]
                                pred_program_scores, _ = fsq.decode(padded_latents.float()) # [b, seqlen, num_levels]
                                pred_program_weights = torch.argmax(pred_program_scores, dim=-1).long() # [b, seqlen]

                                pred_outputs = []
                                for b in range(pred_program_weights.shape[0]):
                                    set_weights(program_net, pred_program_weights[b], pn_interval, pn_lower_limit)
                                    outputs = program_net.predict(target_inputs[b]) # [num_examples, seqlen]
                                    pred_outputs.append(outputs)
                                pred_outputs = torch.stack(pred_outputs).long().to(device) # [b, num_examples, seqlen]

                                # prepare fantasy items and add to fantasy dataset
                                for i in range(len(minibatch)):
                                    item = {}
                                    item_examples = torch.cat([target_inputs[i], pred_outputs[i]], dim=-1) # [num_examples, seqlen * 2]
                                    item_program = pred_program_weights[i] # [seqlen]
                                    item['examples'] = item_examples.cpu()
                                    item['program'] = item_program.cpu()
                                    item['qid'] = 'fantasy'
                                    new_fantasy_dataset.append(item)


            ## back to sleep mode training

            # alternatively train on train_dataset and fantasy_dataset
            sleep_on_what = -1
            if (len(train_dataset) > 0) or (len(new_fantasy_dataset) > 0):
                
                if len(train_dataset) > 0:
                    sleep_dataset = train_dataset 
                    sleep_on_what = 0
                if (len(new_fantasy_dataset) > 0):
                    if train_step % 2 == 0:
                        sleep_dataset = new_fantasy_dataset
                        sleep_on_what = 1
                        if (len(all_fantasy_dataset) > 0) and (train_step % 4 == 0):
                            sleep_dataset = all_fantasy_dataset
                            sleep_on_what = 2

                if sleep_on_what != -1:

                    # fetch minibatch
                    idx = np.arange(len(sleep_dataset))
                    np.random.shuffle(idx)
                    idx = idx[:batch_size]
                    minibatch = [sleep_dataset[i] for i in idx]

                    # prepare programs 
                    programs = []
                    for item in minibatch:
                        programs.append(item['program'])
                    programs = torch.stack(programs).to(device) # [b, seqlen]

                    # get example embeddings
                    with torch.no_grad():
                        example_embs = example_embedder(minibatch) # [b, seqlen, example_emb_dim]

                    # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
                    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                    # forward prop through FSQ 
                    recon_program_scores, z_e, z_q, usage, unique, percent_masked_latents = fsq(programs) # recon_programs.shape: [b, seq_len, num_levels]

                    # calculate loss
                    recon_loss = reconstruction_factor * reconstruction_loss(recon_program_scores, programs)
                    compress_loss = compress_factor * compression_loss(z_e)
                    loss = recon_loss + compress_loss 

                    loss.backward()
                    # gradient cliping 
                    torch.nn.utils.clip_grad_norm_(sleep_params, max_norm=1.0)
                    # gradient step
                    sleep_optimizer.step()
                    sleep_optimizer.zero_grad()

                    # bookeep losses
                    if sleep_on_what == 1:
                        results_train_loss = ema(results_train_loss, loss.item())
                        results_codebook_usage.append(usage.item())
                        results_codebook_unique.append(unique)
                        results_masked_latents.append(percent_masked_latents.item())
                        results_recon_loss.append(recon_loss.item())
                        results_compress_loss.append(compress_loss.item())

                    if len(results_train_loss) > 0:
                        pbar.set_description('mode:{} loss: {:.3f}'.format(sleep_mode, results_train_loss[-1]))

            pbar.update(1)

            dit.train()
            example_embedder.train()


        if sleep_mode == 2: # dream mode - train DiT
            fsq.eval()
            # example_embedder.eval()

            ## dream mode training

            # alternatively train on replays and fantasies 
            dream_on_what = -1
            if (len(train_dataset) > 0) or (len(new_fantasy_dataset) > 0):

                if len(train_dataset) > 0:
                    dream_dataset = train_dataset 
                    dream_on_what = 0
                if (len(new_fantasy_dataset) > 0):
                    if train_step % 2 == 0:
                        dream_dataset = new_fantasy_dataset 
                        dream_on_what = 1
                        if (len(all_fantasy_dataset) > 0) and (train_step % 4 == 0):
                            dream_dataset = all_fantasy_dataset
                            dream_on_what = 2

                if dream_on_what != -1:

                    # fetch minibatch
                    idx = np.arange(len(dream_dataset))
                    np.random.shuffle(idx)
                    idx = idx[:batch_size]
                    minibatch = [dream_dataset[i] for i in idx]

                    # prepare programs
                    programs = []
                    for item in minibatch:
                        programs.append(item['program'])
                    programs = torch.stack(programs).to(device) # [b, seqlen]

                    # get example embeddings
                    # with torch.no_grad():
                    example_embs = example_embedder(minibatch) # [b, seqlen, example_emb_dim]

                    ## loss for DiT 

                    # forward prop through fsq encoder to get target latents
                    with torch.no_grad():
                        z_e = fsq.encode(programs) # z_e.shape: [b, seq_len,  img_latent_dim]
                        latents, _, _, _, _, target_idx = fsq.quantize(z_e) # target_idx.shape: [b * img_latent_seqlen]
                        target_idx = target_idx.view(-1, program_seq_len) # [b, seqlen] 

                    target_idx = target_idx[:, :program_latent_seqlen] # program latent slicing

                    x = target_idx # x.shape: [b, seq_len] 
                    condition = example_embs

                    # set condition = None with prob p_uncond
                    if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
                        condition = None

                    # sample diffusion time ~ uniform(eps, 1)
                    t = (1 - diffusion_start_time_eps) * torch.rand(x.shape[0], device=device) + diffusion_start_time_eps

                    # get noise from noise schedule
                    sigma, dsigma = logLinearNoise(t)

                    # perturb the data
                    x_perturb = perturb(x, sigma, dit_mask_token)

                    # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
                    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                    # get score
                    log_score = dit(x_perturb, sigma, condition)

                    # calculate loss 
                    dit_loss = score_entropy_loss(log_score, sigma.unsqueeze(-1), x_perturb, x, dit_mask_token)
                    dit_loss = (dsigma.unsqueeze(-1) * dit_loss).sum(dim=-1).mean()

                    ## total dream loss 
                    dream_loss = dit_loss 

                    dream_loss.backward()
                    # gradient cliping - helps to prevent unnecessary divergence 
                    torch.nn.utils.clip_grad_norm_(dream_params, max_norm=1.0)
                    # gradient step
                    dream_optimizer.step()
                    dream_optimizer.zero_grad()

                    # bookeep losses
                    if dream_on_what == 1:
                        results_dit_loss = ema(results_dit_loss, dit_loss.item())

                    if len(results_dit_loss) > 0:
                        pbar.set_description('mode:{} loss: {:.2f}'.format(sleep_mode, results_dit_loss[-1]))

            pbar.update(1)

            fsq.train()
            # example_embedder.train()


        ## sampling
        # if (train_step+1) % sampling_freq == 0:
        if sleep_mode_counter == 1:

            # visualize wake dataset 
            if len(wake_visualize_dataset) > 0:
                savepath = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0)
                visualize_grids(wake_visualize_dataset[:1], savepath, square_dim)
                wake_visualize_dataset = []


            if sleep_mode == 1: # sleep mode - eval FSQ 

                # visualize reconstructions
                if sleep_on_what != -1:

                    example = minibatch[0]['examples'][0]
                    example_input, example_output = example[:(square_dim**2)], example[-(square_dim**2):] # [seqlen]

                    recon_program_scores = recon_program_scores[0].detach() # [seqlen, num_levels]
                    pred_program_weights = torch.argmax(recon_program_scores, dim=-1).long() # [seqlen]

                    # forward prop through program net to get pred outputs 
                    set_weights(program_net, pred_program_weights, pn_interval, pn_lower_limit)
                    pred_output = program_net.predict(example_input.unsqueeze(0).to(device)).long() # [1, seqlen]
                    pred_output = pred_output.squeeze(0).cpu() # [seqlen]

                    vis_item = {}
                    vis_item['pred_answer'] = pred_output 
                    vis_item['query'] = example_input 
                    vis_item['answer'] = example_output 
                    vis_item['qid'] = minibatch[0]['qid']
                    savepath = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(1)
                    visualize_grids([vis_item], savepath, square_dim)


            if sleep_mode == 2: # dream mode - eval DIT 
            
                # put model in eval mode to avoid dropout
                dit.eval()
                fsq.eval()
                example_embedder.eval()

                # generate sample from dit
                # if dream_on_what != -1:
                if sleep_on_what != -1:

                    with torch.no_grad():

                        # prepare target inputs and outputs 
                        target_inputs, target_outputs = [], []
                        for item in minibatch:
                            example_inputs, example_outputs = item['examples'][:, :(square_dim**2)], item['examples'][:, -(square_dim**2):]
                            target_inputs.append(example_inputs)
                            target_outputs.append(example_outputs)
                        target_inputs, target_outputs = torch.stack(target_inputs).to(device), torch.stack(target_outputs).to(device) # [b, num_examples, seqlen]

                        # get sample tokens corresponding to indices of codebook 
                        x_sample = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, example_embs.shape[0], example_embs, cfg_scale, sampling_eps, device) # x_sample.shape: [b, seqlen]
                        x_sample = x_sample.flatten() # shape: [b * seqlen]

                        # get codebook vectors indexed by x_sample
                        sampled_latents = fsq.codebook[x_sample] # sampled_latents.shape: [b, seqlen, latent_dim]
                        sampled_latents = sampled_latents.unflatten(dim=0, sizes=(-1, program_latent_seqlen)) # [b, program_latent_seqlen, latent_dim]
                        padded_latents = torch.zeros(sampled_latents.shape[0], program_seq_len, sampled_latents.shape[-1], device=sampled_latents.device)
                        padded_latents[:, :sampled_latents.shape[1]] = sampled_latents
                        padded_latents = padded_latents.flatten(start_dim=0, end_dim=1) # [b * program_seq_len, latent_dim]
                        # get codebook vectors indexed by x_sample
                        pred_program_scores, _ = fsq.decode(padded_latents.float()) # [b, seqlen, num_levels]
                        pred_program_weights = torch.argmax(pred_program_scores, dim=-1).long() # [b, seqlen]

                        # forward prop through program net to get pred outputs 
                        pred_outputs = []
                        for b in range(pred_program_weights.shape[0]):
                            set_weights(program_net, pred_program_weights[b], pn_interval, pn_lower_limit)
                            outputs = program_net.predict(target_inputs[b]) # [num_examples, seqlen]
                            pred_outputs.append(outputs)
                        pred_outputs = torch.stack(pred_outputs).long().to(device) # [b, num_examples, seqlen]

                        # get solved minibatch idx
                        solved_minibatch_idx = torch.where((pred_outputs == target_outputs).all(dim=(-1, -2)))[0]
                        dit_batch_accuracy = len(solved_minibatch_idx) / example_embs.shape[0]
                        results_dit_batch_accuracy.append(dit_batch_accuracy)

                        # visualize generated sample
                        vis_item = {}
                        vis_item['pred_answer'] = pred_outputs[0][0].cpu()
                        vis_item['query'] = target_inputs[0][0].cpu()
                        vis_item['answer'] = target_outputs[0][0].cpu()
                        vis_item['qid'] = minibatch[0]['qid']
                        savepath = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2)
                        visualize_grids([vis_item], savepath, square_dim)

                dit.train()
                fsq.train()
                example_embedder.train()


        ## plotting
        # if (train_step+1) % plot_freq == 0: ## save ckpt and plot losses
        if sleep_mode_counter == 1:


            if sleep_mode != 0: # sleep mode - plot for FSQ

                # wake mode plots shifted to sleep mode
                if len(results_test_items) > 0:

                    fig = plt.figure()
                    plt.plot(results_test_items, label='test_items')
                    plt.legend()
                    plt.title('val:{}'.format(results_test_items[-1]))
                    save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0) + '_remItems.png'
                    fig.savefig(save_path)
                    plt.close(fig)

                    if len(results_wake_pixel_accuracy) > 0:
                        fig = plt.figure()
                        plt.plot(results_wake_pixel_accuracy, label='pixel_accuracy')
                        plt.plot(results_wake_pair_accuracy, label='pair_accuracy')
                        plt.legend()
                        plt.title('pixel:{:.3f} pair:{:.3f}'.format(results_wake_pixel_accuracy[-1], results_wake_pair_accuracy[-1]))
                        save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0) + '_accuracies.png'
                        fig.savefig(save_path)
                        plt.close(fig)


                # plot sleep results
                if len(results_train_loss) > 0:

                    fig, ax = plt.subplots(2,2, figsize=(15,10))

                    ax[0,0].plot(results_recon_loss, label='recon_loss')
                    ax[0,0].plot(results_compress_loss, label='compress_loss')
                    ax[0,0].plot(results_train_loss, label='train_loss')
                    ax[0,0].legend()
                    ax[0,0].set(xlabel='eval_iters')
                    ax[0,0].set_title('train:{:.3f} recon:{:.3f} compress:{:.3f}'.format(results_train_loss[-1], results_recon_loss[-1], results_compress_loss[-1]))

                    ax[1,0].plot(results_codebook_unique, label='codebook_unique')
                    ax[1,0].legend()
                    ax[1,0].set(xlabel='eval_iters')
                    ax[1,0].set_title('val:{:.3f}'.format(results_codebook_unique[-1]))

                    ax[0,1].plot(results_codebook_usage, label='codebook_usage')
                    ax[0,1].legend()
                    ax[0,1].set(xlabel='train_iters')
                    ax[0,1].set_title('val:{:.3f}'.format(results_codebook_usage[-1]))

                    ax[1,1].plot(results_masked_latents, label='percent_masked_latents')
                    ax[1,1].legend()
                    ax[1,1].set(xlabel='train_iters')
                    ax[1,1].set_title('val:{:.3f}'.format(results_masked_latents[-1]))

                    # plt.suptitle('final_train_loss: ' + str(results_train_loss[-1]))
                    save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(1) + '.png'
                    plt.savefig(save_path)
                    plt.close(fig)


            if sleep_mode != 0: # 2: # dream mode - plot for DIT 

                # plot dit loss
                if len(results_dit_loss) > 0:

                    fig = plt.figure()
                    plt.plot(results_dit_loss, label='dit_loss')
                    plt.legend()
                    plt.title('final_loss:{:.3f}'.format(results_dit_loss[-1]))
                    plt.ylim([0, 100])
                    save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2) + '_ditLoss.png'
                    fig.savefig(save_path)
                    plt.close(fig)

                # plot dit batch_accuracy
                if len(results_dit_batch_accuracy) > 0:

                    fig = plt.figure()
                    plt.plot(results_dit_batch_accuracy, label='dit_batch_accuracy')
                    plt.legend()
                    plt.title('final_loss:{:.3f}'.format(results_dit_batch_accuracy[-1]))
                    save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2) + '_ditAccuracy.png'
                    fig.savefig(save_path)
                    plt.close(fig)


        train_step += 1
        sleep_mode_counter += 1

    pbar.close()
import torch
from torch.utils.data import DataLoader
from utils import memory, memory_rotta
from utils.loss_functions import *
from utils.sam_optimizer import SAM, sam_collect_params
import time

class SoTTA(nn.Module):

    def __init__(self, model, steps, args):
        super().__init__()

        # turn on grad for BN params only
        self.net = model
        self.steps = steps
        # self.device = torch.device("cuda:{:d}".format(args.gpu_idx) if torch.cuda.is_available() else "cpu")
        params, _ = sam_collect_params(self.net, freeze_top=True)
        self.optimizer = SAM(params, torch.optim.Adam, rho=0.05, lr=args.lr_online,
                                weight_decay=args.weight_decay)
        

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = args.bn_momentum  # use the online adaptation for BN parameters
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d): 
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):  
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        # self.fifo = memory.FIFO(capacity=args.update_every_x)  # required for evaluation
        
        # Initialize memory based on the specified type for online learning
        if args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=args.memory_size)
        elif args.memory_type == 'HUS':
            self.mem = memory.HUS(capacity=args.memory_size, class_num=args.class_num, threshold=args.high_threshold)
        elif args.memory_type == 'CSTU':
            self.mem = memory_rotta.CSTU(capacity=args.memory_size, num_class=args.opt['num_class'],
                                         lambda_t=1, lambda_u=1)  # replace memory with original RoTTA
        elif args.memory_type == 'ConfFIFO':
            self.mem = memory.ConfFIFO(capacity=args.memory_size, threshold=args.high_threshold)
        
        self.mem_state = self.mem.save_state_dict()

        self.ema = None
        self.batchnorm_stats = []

    def train_online(self, current_num_sample, current_sample, args, add_memory=True, evaluation=True):
        """
        Train the model online 
        """

        # In sotta, every online coming sample will be choosen to be stored in the memory
        if add_memory:
            # self.fifo.add_instance(current_sample)  # for batch-based inference

            with torch.no_grad():

                self.net.eval()

                if args.memory_type in ['FIFO']:
                    self.mem.add_instance(current_sample)

                elif args.memory_type in ['HUS', 'ConfFIFO']:
                    f, c, d = current_sample.cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
                    _, logit = self.net(f)
                    pseudo_cls = logit.max(1, keepdim=False)[1][0].cpu().numpy()
                    pseudo_conf = F.softmax(logit, dim=1).max(1, keepdim=False)[0][0].cpu().numpy()
                    self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])

                elif args.memory_type in ['CSTU']:
                    f, c, d = current_sample.cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
                    _, ema_out = self.net(f)
                    predict = torch.softmax(ema_out, dim=1)
                    pseudo_label = torch.argmax(predict, dim=1)
                    entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

                    for i, data in enumerate(f):
                        p_l = pseudo_label[i].item()
                        uncertainty = entropy[i].item()
                        current_instance = (data, p_l, uncertainty)
                        self.mem.add_instance(current_instance)

                else:
                    raise NotImplementedError

        # if not in the adaptation interval, it will only continue to evaluate the model without updating the weights
        if evaluation:
            _, outputs = self.net(current_sample)

        # if it is the adaptation interval, it will try to update the model 
        # following the original paper of sotta, the adaptation interval is the same as the memory capacity
        if current_num_sample % args.update_every_x == 0: 
            update_start_time = time.time()
            # setup models
            self.net.train()

            if args.memory_type in ['CSTU']:
                feats, _ = self.mem.get_memory()
            else:
                feats, _, _ = self.mem.get_memory()

            print("current sample index for updating: {}, memory size: {}".format(current_num_sample, len(feats)))

            if len(feats)==0:
                return outputs # if no memory to train on, return current outputs directly

            feats = torch.stack(feats)
            dataset = torch.utils.data.TensorDataset(feats)
            data_loader = DataLoader(dataset, batch_size=args.batch_size_online,
                                    shuffle=True, drop_last=False, pin_memory=False)

            entropy_loss = HLoss(args.temperature)

            for e in range(self.steps):
                for batch_idx, (feats,) in enumerate(data_loader):
                    self.step(loss_fn=entropy_loss, feats=feats)
            update_end_time = time.time()
            print("Adaptation time: {:.3f} seconds.".format(update_end_time - update_start_time))
        return outputs

    def step(self, loss_fn, feats=None):
        assert (feats is not None)

        self.net.train()
        
        feats = feats.squeeze(1).cuda()

        _, preds_of_data = self.net(feats)

        loss_first = loss_fn(preds_of_data)

        self.optimizer.zero_grad()

        loss_first.backward()

        if not isinstance(self.optimizer, SAM):
            self.optimizer.step()
        else:
            # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            self.optimizer.first_step(zero_grad=True)

            _, preds_of_data = self.net(feats)

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second = loss_fn(preds_of_data)

            loss_second.backward()

            self.optimizer.second_step(zero_grad=True)
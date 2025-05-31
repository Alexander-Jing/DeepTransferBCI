from torch.utils.data import DataLoader

import conf
from utils import memory, memory_rotta
from utils.loss_functions import *
from utils.sam_optimizer import SAM
from .dnn import DNN

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SoTTA(nn.Module):

    def __init__(self, model, optimizer, steps, args):
        super().__init__()

        # turn on grad for BN params only
        self.net = model
        self.optimizer = optimizer
        self.steps = steps
       
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

        self.fifo = memory.FIFO(capacity=args.update_every_x)  # required for evaluation
        
        # Initialize memory based on the specified type for online learning
        if args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=args.memory_size)
        elif args.memory_type == 'HUS':
            self.mem = memory.HUS(capacity=args.memory_size, threshold=args.high_threshold)
        elif args.memory_type == 'CSTU':
            self.mem = memory_rotta.CSTU(capacity=args.memory_size, num_class=args.opt['num_class'],
                                         lambda_t=1, lambda_u=1)  # replace memory with original RoTTA
        elif args.memory_type == 'ConfFIFO':
            self.mem = memory.ConfFIFO(capacity=args.memory_size, threshold=args.high_threshold)
        
        self.mem_state = self.mem.save_state_dict()

        self.ema = None
        self.batchnorm_stats = []

    def train_online(self, current_num_sample, args, add_memory=True, evaluation=True):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        # Get a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        
        # Add into memory
        if add_memory:
            self.fifo.add_instance(current_sample)  # for batch-based inference

            with torch.no_grad():

                self.net.eval()

                if args.memory_type in ['FIFO']:
                    self.mem.add_instance(current_sample)

                elif args.memory_type in ['HUS', 'ConfFIFO']:
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                    logit = self.net(f.unsqueeze(0))
                    pseudo_cls = logit.max(1, keepdim=False)[1][0].cpu().numpy()
                    pseudo_conf = F.softmax(logit, dim=1).max(1, keepdim=False)[0][0].cpu().numpy()
                    self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])

                elif args.memory_type in ['CSTU']:
                    f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                    ema_out = self.net(f.unsqueeze(0))
                    predict = torch.softmax(ema_out, dim=1)
                    pseudo_label = torch.argmax(predict, dim=1)
                    entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

                    for i, data in enumerate(f.unsqueeze(0)):
                        p_l = pseudo_label[i].item()
                        uncertainty = entropy[i].item()
                        current_instance = (data, p_l, uncertainty)
                        self.mem.add_instance(current_instance)

                else:
                    raise NotImplementedError
                
        # 当没有积累满args.update_every_x的数据的时候，仅仅只进行数据存储进入这个memory
        if current_num_sample % args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # 当数据累积满args.update_every_x的数据的时候（更新的间隔达到），也就是累积满一个Online batch的时候，可以进行评估，并且进行更新，文中更新的间隔和memory的大小一致
        if evaluation:
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        if args.memory_type in ['CSTU']:
            feats, _ = self.mem.get_memory()
        else:
            feats, _, _ = self.mem.get_memory()

        if len(feats) == 0:
            return TRAINED

        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)

        entropy_loss = HLoss(args.temperature)

        for e in range(args.epoch):
            for batch_idx, (feats,) in enumerate(data_loader):
                self.step(loss_fn=entropy_loss, feats=feats)

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    def step(self, loss_fn, args, feats=None):
        assert (feats is not None)

        if args.tta_attack_type: # avoid attack error
            feats = feats.clone().detach()

        self.net.train()
        feats = feats.to(device)
        preds_of_data = self.net(feats)

        loss_first = loss_fn(preds_of_data)

        self.optimizer.zero_grad()

        loss_first.backward()

        if not isinstance(self.optimizer, SAM):
            self.optimizer.step()
        else:
            # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            self.optimizer.first_step(zero_grad=True)

            preds_of_data = self.net(feats)

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second = loss_fn(preds_of_data)

            loss_second.backward()

            self.optimizer.second_step(zero_grad=True)
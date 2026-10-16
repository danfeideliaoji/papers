
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
           

import torch
torch.set_num_threads(14)

from torch import nn
import torch.optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

import timm

from losses import SPALoss, SPAInvLink
from utils_metric import mAP, eval_calibration


def main(args):
    # Setup model
    print('creating model...')
    
    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=args.num_classes).cuda()

    if args.model_path:  # make sure to load pretrained ImageNet model
        model.load_state_dict(torch.load(args.model_path), strict=False)

    print('done\n')

    from utils_data import get_Dataset
    train_dataset, val_dataset = get_Dataset(args)

    print(f"len(val_dataset)): {len(val_dataset)}")
    print(f"len(train_dataset)): {len(train_dataset)}")

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # model = nn.DataParallel(model)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args)


def train_multi_label_coco(model, train_loader, val_loader, args):
    criterion = SPALoss(k0=args.k0, b0=args.b0, zeta0=args.zeta0)


    def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    def add_weight_decay_no_fc(model, weight_decay=1e-4, skip_list=(), exclude_list=[]):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights

            if name in exclude_list:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


    if 'ecaresnet50t' in args.model_name:
        parameters = add_weight_decay_no_fc(model, args.weight_decay, exclude_list=['fc.weight', 'fc.bias'])
        parameters_fc = add_weight_decay(model.fc, args.weight_decay)

    elif 'vit_base_patch32_224' in args.model_name:
        parameters = add_weight_decay_no_fc(model, args.weight_decay, exclude_list=['head.weight', 'head.bias'])
        parameters_fc = add_weight_decay(model.head, args.weight_decay)

    else:
        raise ValueError
    
    for para in parameters:
        para['lr'] = args.lr * 0.1

    optimizer = torch.optim.SGD(params = parameters + parameters_fc, lr=args.lr, momentum=0.9, weight_decay=0)  # true wd, filter_bias_and_bn


    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs= args.num_epoches,
                                    pct_start=0.2)
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(args.num_epoches):

        start = time.time()

        if args.train:

            for i, (inputData, target) in enumerate(train_loader):
                inputData = inputData.cuda()
                target = target.cuda()  # (batch,3,num_classes)

                if len(target.shape) == 3:
                    target = target.max(dim=1)[0]
                else:
                    target = target

                with autocast():  # mixed precision
                    output = model(inputData).float() 
                
                loss, los_pos, los_neg = criterion(output, target)
                model.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                # store information
                if i % args.print_freq == 0:
                    trainInfoList.append([epoch, i, loss.item()])
                    print(f'Epoch [{epoch}/{args.num_epoches}], Step [{str(i).zfill(3)}/{str(steps_per_epoch).zfill(3)}], LR {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.1f}')
                    print(f'los_pos: {los_pos.item():.2f}, los_neg: {los_neg.item():.2f}, loss: {loss.item():.2f}')

        model.eval()
        mAP_score = validate_multi(val_loader, model, ema_model=None)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), os.path.join(
                output_dir, 'model-highest.ckpt'))


        print(f'current_mAP = {mAP_score:.2f}, highest_mAP = {highest_mAP:.2f}\n')

        end = time.time()
        print(f'This epoch takes {end - start:.1f} seconds.')


def validate_multi(val_loader, model, ema_model=None):
    print("starting validation")

    preds_regular = []
    targets = []

    invlink = SPAInvLink(k0=args.k0, b0=args.b0, zeta0=args.zeta0)

    for i, (input, target) in enumerate(val_loader):

        if len(target.shape) == 3:
            target = target.max(dim=1)[0]
        else:
            target = target
        
        # compute output
        with torch.no_grad():
            # with autocast(dtype=torch.float32):
            with autocast():
                output_regular = invlink(model(input.cuda())).cpu()


        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())

        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_x_score_regular = mAP(torch.cat(targets).numpy().transpose(), torch.cat(preds_regular).numpy().transpose())

    print(f"mAP@y score regular {mAP_score_regular:.2f}, mAP@x score regular {mAP_x_score_regular:.2f}")


    output_2d = torch.cat(preds_regular)
    target_2d = torch.cat(targets)
    eval_calibration(output_2d, target_2d)

    return mAP_score_regular

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
    parser.add_argument('--data-dir', help='path to dataset', default='/mnt/data/')

    parser.add_argument('--dataset', default='coco', choices=['coco'])

    parser.add_argument('--weight-decay', default=1e-4, type=float) 
    parser.add_argument('--num-epoches', default=40, type=int) 


    # parser.add_argument('--model-name', default='ecaresnet50t.a1_in1k', choices=['ecaresnet50t.a1_in1k', 'vit_base_patch32_224.augreg_in21k'])    
    # parser.add_argument('--lr', default=1e-3, type=float) # ecaresnet


    parser.add_argument('--model-name', default='vit_base_patch32_224', choices=['ecaresnet50t', 'vit_base_patch32_224'])
    # parser.add_argument('--lr', default=1e-4, type=float) # VIT
    parser.add_argument('--lr', default=5e-5, type=float) # VIT


    parser.add_argument('--k0', default=3, type=float)
    parser.add_argument('--b0', default=1.0, type=float)
    parser.add_argument('--zeta0', default=5, type=float)

    # parser.add_argument('--k0', default=1, type=float)
    # parser.add_argument('--b0', default=0.0, type=float)
    # parser.add_argument('--zeta0', default=1, type=float)


    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--pretrained', default=True, type=bool)

    parser.add_argument('--model-path', default='', type=str)

    parser.add_argument('--num-classes', default=80)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--image-size', default=224, type=int,
                        metavar='N', help='input image size (default: 224)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    parser.add_argument('--print-freq', '-p', default=256, type=int,
                        metavar='N', help='print frequency (default: 256)')

    args = parser.parse_args()

    # get current time
    import datetime, pytz, time, sys, socket
    pst = pytz.timezone('America/Los_Angeles')
    ct = datetime.datetime.now(pst).strftime("%Y-%m-%d-%H:%M:%S.%f")

    if args.train:
        output_dir = os.path.join('runs', 'train', args.dataset, args.model_name, socket.gethostname(), ct)
    else:
        output_dir = os.path.join('runs', 'val', args.dataset, args.model_name, socket.gethostname(), ct)

    # os.mkdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from utils_logger import Logger
    log_file = os.path.join(output_dir, 'log.log')
    err_file = os.path.join(output_dir, 'err.log')

    sys.stdout = Logger(log_file)
    sys.stderr = Logger(err_file)  # redirect std err, if necessary

    from pprint import pprint
    pprint(vars(args))

    assert torch.cuda.is_available()
    print(f'There are {torch.cuda.device_count()} available GPUs!') 

    if 'ecaresnet50t' in args.model_name:
        args.model_name = 'ecaresnet50t.a1_in1k'
    elif 'vit_base_patch32_224' in args.model_name:
        args.model_name = 'vit_base_patch32_224.augreg_in21k'
    else:
        raise ValueError
    
    main(args)

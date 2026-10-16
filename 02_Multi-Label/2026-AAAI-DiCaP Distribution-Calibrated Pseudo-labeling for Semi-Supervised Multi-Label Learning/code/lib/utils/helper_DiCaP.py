from lib.utils.helper import function_mAP
import torch
import os
import numpy as np

@torch.no_grad()
def get_output(loader, model, args):

    model.eval()
    import numpy as np
    true_label = []
    outputs = []
    outputs_ori_list = []
    targets_list = []

    for i, ((inputs_list, _), targets, index) in enumerate(loader):

        inputs_ori = inputs_list[0].cuda(non_blocking=True)
        
        targets = targets.cuda(non_blocking=True)
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori = model(inputs_ori)
                    
        outputs_ori_ = outputs_ori
        

        outputs_ori_sm = torch.sigmoid(outputs_ori_)
        
        # add list
        outputs_ori_list.append(outputs_ori_sm.detach().cpu())        
        targets_list.append(targets.detach().cpu())

        outputs.append(outputs_ori_sm.cpu().numpy())
        true_label.append(targets.cpu().numpy())


    labels = np.concatenate(targets_list)
    outputs_ori = np.concatenate(outputs_ori_list)

    # calculate mAP
    mAP_ori = function_mAP(labels, outputs_ori)

    outputs = np.vstack(outputs)      # [total_samples, num_classes]
    true_label = np.vstack(true_label)        # [total_samples, num_classes]

    return outputs, true_label, mAP_ori

def save_data_in_train(lb_validate_loader, ul_loader, model, args, logger, epoch):

    outputs_lb, label_lb, mAP_lb = get_output(lb_validate_loader, model, args)
    logger.info("  mAP_lb: {}".format(mAP_lb))  
    outputs_ul, label_ul, mAP_ul = get_output(ul_loader, model, args)
    logger.info("  mAP_ul: {}".format(mAP_ul))
    
    save_path = f'{args.output}/distribution/'
    os.makedirs(save_path, exist_ok=True)

    save_path_lb = os.path.join(save_path , f'{epoch}.pth')

    torch.save({
        'label_lb': label_lb,
        'outputs_lb': outputs_lb,
        'label_ul': label_ul,
        'outputs_ul': outputs_ul,
    }, save_path_lb)


def smooth_with_outlier_removal_and_polynomial(data, centers, outlier_threshold=1.5):
    
    valid_indices = ~np.isnan(data) & (data >= 0)
    if np.sum(valid_indices) < 3:
        return data
    
    valid_data = data[valid_indices]
    valid_centers = np.array(centers)[valid_indices]
    
    
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_threshold * IQR
    upper_bound = Q3 + outlier_threshold * IQR
    
    # 过滤离群点
    outlier_mask = (valid_data >= lower_bound) & (valid_data <= upper_bound)
    outlier_mask[-1] = True 
    outlier_mask[0] = True  
    outlier_mask[1] = True 
    valid_data[0] = 0
    valid_data[1] = 0
    clean_data = valid_data[outlier_mask]
    clean_centers = valid_centers[outlier_mask]
    
    if len(clean_data) < 3:
        return data
    
    
    poly_degree = min(3, len(clean_data) - 1)
    coeffs = np.polyfit(clean_centers, clean_data, poly_degree)
    poly_func = np.poly1d(coeffs)
    
    smooth_values = poly_func(centers)
    
    smooth_values = np.clip(smooth_values, 0, 1)
    
    return smooth_values

def smooth_for_voc(data_ori, centers, num_bins):
   
    if num_bins >= 3:        
        data_smooth0 = data_ori.copy()
        for i in range(1, num_bins-1):
            if data_ori[i] == 1:
                data_smooth0[i] = (data_ori[i-1]  + data_ori[i+1]) / 2
        for i in range(3, num_bins-1):
            if data_ori[i] == 0:
                data_smooth0[i] = (data_ori[i-1]  + data_ori[i+1]) / 2
        data_smooth1 = np.zeros_like(data_ori)
        data_smooth1[0] = data_ori[0]
        data_smooth1[-1] = data_ori[-1]
        for i in range(1, num_bins-1):
            data_smooth1[i] = (data_smooth0[i-1] + data_smooth0[i] + data_smooth0[i+1]) / 3
    else:
        data_smooth1 = data_ori.copy()
    
    data_smooth2 = data_smooth1.copy()
    for i in range(num_bins-2, -1, -1):
        if data_smooth2[i] > data_smooth2[i+1]:
            data_smooth2[i] = data_smooth2[i+1]
    for i in range(1, num_bins):
        if data_smooth2[i] < data_smooth2[i-1]:
            data_smooth2[i] = data_smooth2[i-1]

    data_smooth_last = smooth_with_outlier_removal_and_polynomial(
        data_smooth2, centers, outlier_threshold=1.5
    )
    data_smooth_last[-1] = data_ori[-1]
    return data_smooth_last

def smooth_for_large_dataset(data_ori, centers, num_bins):
   
    if num_bins >= 3:
        data_smooth0 = data_ori.copy()
        for i in range(1, num_bins-1):
            if data_ori[i] == 1:
                data_smooth0[i] = (data_ori[i-1]  + data_ori[i+1]) / 2
        for i in range(3, num_bins-1):
            if data_ori[i] == 0:
                data_smooth0[i] = (data_ori[i-1]  + data_ori[i+1]) / 2
        data_smooth1 = np.zeros_like(data_ori)
        data_smooth1[0] = data_ori[0]
        data_smooth1[-1] = data_ori[-1]
        for i in range(1, num_bins-1):
            data_smooth1[i] = (data_smooth0[i-1] + data_smooth0[i] + data_smooth0[i+1]) / 3
    else:
        data_smooth1 = data_ori.copy()
    
    data_smooth2 = data_smooth1.copy()
    for i in range(num_bins-2, -1, -1):
        if data_smooth2[i] > data_smooth2[i+1]:
            data_smooth2[i] = data_smooth2[i+1]
    for i in range(1, num_bins):
        if data_smooth2[i] < data_smooth2[i-1]:
            data_smooth2[i] = data_smooth2[i-1]

    data_smooth2[-1] = data_ori[-1]
    return data_smooth2

"""固定已训练模型，检查训练/dev 泛化差距、图编号敏感性及训练接线。

不训练、不调参、不计算新的测试指标；所有模型导入来自已完成实验的源码快照。
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='results_paper_aligned_5fold_fold1_80epochs')
    cli=parser.parse_args()
    run=Path(cli.run).resolve()
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==h,name
    sys.path.insert(0,str(run/'source'))
    from strict_training import VAMS,transform,metrics,knn_indices,label_frequency_loss
    import demo
    config=SimpleNamespace(**manifest['args']); demo.args=config
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    device=torch.device('cpu')
    views,labels=demo.data_loader(str(Path(config.data_root)/config.dataset))
    labels=torch.as_tensor(labels).float()
    if len(labels)!=len(views[0]): labels=labels.T
    checkpoint=torch.load(run/'checkpoints/fold_1.pth',map_location='cpu',weights_only=False)
    model=VAMS(checkpoint['input_dims'],checkpoint['labels'],config).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    train_ids,dev_ids=(checkpoint['split'][k] for k in ['train','dev'])
    train=transform(views,train_ids,checkpoint['scalers'],device)
    dev=transform(views,dev_ids,checkpoint['scalers'],device)
    with torch.no_grad():
        reference=model.encode(train); query=model.encode(dev)
        train_neighbors=knn_indices(reference,reference,config.k,True)
        dev_neighbors=knn_indices(reference,query,config.k)
        def predict(encoded,neighbors):
            return torch.cat([model.predict_batch(encoded[s:s+config.batch_size],reference,neighbors,b,config)
                for b,s in enumerate(range(0,len(encoded),config.batch_size))])
        train_scores=predict(reference,train_neighbors)
        dev_scores=predict(query,dev_neighbors)
        train_metrics=metrics(labels[train_ids],train_scores,config.threshold)
        dev_metrics=metrics(labels[dev_ids],dev_scores,config.threshold)
        baseline_identity=model.label_identity.detach().clone()
        # 节点排列变化，身份向量不变；输出恢复标签顺序后应代表同一语义预测。
        permutations=[]
        for permutation in [list(reversed(range(model.labels))),list(range(1,model.labels))+[0]]:
            model.label_identity=baseline_identity[permutation]
            reordered=predict(query,dev_neighbors)[:,np.argsort(permutation)]
            delta=(reordered-dev_scores).abs()
            permutations.append({'permutation':permutation,'max_score_change':float(delta.max()),
                'mean_score_change':float(delta.mean()),
                'threshold_positions_changed':int(((reordered>=.5)!=(dev_scores>=.5)).sum()),
                'label_order_rows_changed':int((torch.argsort(reordered,dim=1)!=torch.argsort(dev_scores,dim=1)).any(1).sum())})
        model.label_identity=baseline_identity
        # 在边更新器上，单纯交换一条无向边的端点索引，不改变任何节点身份或旧边属性。
        embedded_labels=model.label_projection(model.label_identity)
        placeholder=torch.zeros(model.views,config.batch_size,model.labels)
        gi,gl=model.graph.graph_generator(query[:config.batch_size],embedded_labels,placeholder,config,dev_neighbors,reference,0)
        gc=model.graph.generate_cross_graph(gl,torch.tensor(gi.kwargs),config.k)
        si=(gi.x,gi.edge_index,model.graph.red_edge_ins(gi.edge_attr))
        sl=(gl.x,gl.edge_index,model.graph.red_edge_label(gl.edge_attr))
        sc=(gc.edge_index,model.graph.red_edge_cross(gc.edge_attr))
        node=model.graph.layers[0].node_model; node.num_views=model.views; node.num_neighbors=config.k
        ni,nl,nc=node(si,sl,sc)
        edge=model.graph.layers[0].edge_model
        ordinary=edge(ni,nl,nc)[1][2]
        flipped=edge(ni,(nl[0],nl[1].flip(0),nl[2]),nc)[1][2]
        endpoint_difference=float((ordinary-flipped).abs().max())
        # 分开记录四路消息尺度，不能仅因标签通道求和便认定它造成差距。
        h=128; captured=[]
        handle=node.node_mlp_ins.register_forward_pre_hook(lambda m,args: captured.append(args[0].detach()))
        node(si,sl,sc); handle.remove()
        parts=captured[0].split(h,dim=1)
        message_norms={name:float(x.norm(dim=1).mean()) for name,x in zip(['self','other_views','knn','labels'],parts)}
    with (run/'epoch_metrics.csv').open(encoding='utf-8-sig') as f: epochs=list(csv.DictReader(f))
    batches=[labels[train_ids[s:s+config.batch_size]].mean(0).tolist() for s in range(0,len(train_ids),config.batch_size)]
    p=[labels[ids].mean(0).tolist() for ids in [train_ids,dev_ids]]
    train_lr=metrics(labels[train_ids],train_scores,.5)['LRAP']
    hubness=[]
    for v in range(model.views):
        degree=torch.bincount(train_neighbors[v].reshape(-1),minlength=len(train_ids))
        hubness.append({'view':v+1,'max_times_selected':int(degree.max()),
            'unused_reference_rows':int((degree==0).sum()),'top5_incoming_share':float(degree.topk(5).values.sum()/degree.sum())})
    out=Path('diagnosis_paper_run'); out.mkdir(exist_ok=True)
    report={'scope':'saved_model_train_dev_only_no_training','source_snapshot_verified':True,
        'selected_epoch':checkpoint['epoch'],'train_metrics_excluding_self_neighbors':train_metrics,
        'dev_metrics':dev_metrics,'LRAP_generalization_gap':train_lr-dev_metrics['LRAP'],
        'train_weighted_BCE':float(label_frequency_loss(train_scores,labels[train_ids],checkpoint['frequency'],config.beta)),
        'dev_weighted_BCE':float(label_frequency_loss(dev_scores,labels[dev_ids],checkpoint['frequency'],config.beta)),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'dropout_modules':sum(isinstance(m,torch.nn.Dropout) for m in model.modules()),
        'requested_dropout':config.dropout,'label_permutation':permutations,
        'edge_endpoint_swap_max_change':endpoint_difference,'first_layer_message_mean_L2_norm':message_norms,
        'learning_rates':{str(e):config.lr*.933**(e-1) for e in [1,33,80]},
        'late_dev_LRAP':float(epochs[-1]['dev_LRAP']),'late_online_train_loss':float(epochs[-1]['train_loss']),
        'batch_label_frequency':batches,'train_label_frequency':p[0],'dev_label_frequency':p[1],
        'knn_train_hubness':hubness,'test_evaluations':0}
    (out/'inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

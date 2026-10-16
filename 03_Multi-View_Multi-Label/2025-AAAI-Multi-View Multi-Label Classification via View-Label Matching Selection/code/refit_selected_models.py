"""内部验证选择完成后，在完整外层训练折重新拟合；固定配置和轮次，不搜索。"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROTOCOL = 'vams_v5_deterministic_outer_train_refit'


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', default='results_scene_v5_deterministic_search')
    p.add_argument('--output_dir', default='results_scene_v5_refit_outer_train')
    cli = p.parse_args()
    parent = Path(cli.parent).resolve()
    old = json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    assert old['status']=='complete'
    assert json.loads((parent/'verification.json').read_text())['status']=='PASS'
    with (parent/'candidate_metrics.csv').open(encoding='utf-8-sig') as f: trials=list(csv.DictReader(f))
    winners = [max([t for t in trials if int(t['fold'])==i], key=lambda t: float(t['dev_LRAP'])) for i in range(1,6)]
    root = Path(cli.output_dir).resolve(); root.mkdir(exist_ok=False)
    (root/'source').mkdir(); (root/'checkpoints').mkdir()
    hashes = {}
    for name,h in old['source_sha256'].items():
        content=(parent/'source'/name).read_bytes()
        assert hashlib.sha256(content).hexdigest()==h
        (root/'source'/name).write_bytes(content); hashes[name]=h
    content=Path(__file__).read_bytes()
    (root/'source'/'refit_selected_models.py').write_bytes(content)
    hashes['refit_selected_models.py']=hashlib.sha256(content).hexdigest()
    sys.path.insert(0,str(root/'source'))
    from runtime_config import configure_determinism, runtime_metadata
    configure_determinism()
    import torch
    import numpy as np
    import demo
    from strict_training import (VAMS, seed_all, make_splits, fit_scalers, transform, knn_indices,
                                 label_frequency_loss, predict, metrics, save_csv)
    from deterministic_search import state_hash
    args=SimpleNamespace(**old['args']); args.output_dir=str(root)
    data=Path(args.data_root)/args.dataset
    assert hashlib.sha256(data.read_bytes()).hexdigest()==old['data_sha256']
    demo.args=args
    raw,y=demo.data_loader(str(data)); raw=[v.float() for v in raw];y=torch.as_tensor(y).float()
    if y.shape[0]!=len(raw[0]):y=y.T
    splits=make_splits(len(y),args.k_fold,args.seed,args.dev_fraction)
    parent_splits=json.loads((parent/'splits.json').read_text())
    plan=[{'fold':i+1,'candidate':t['candidate'],'epochs':int(t['selected_epoch']),
           'selection_dev_LRAP':float(t['dev_LRAP']),'initial_state_sha256':t['initial_state_sha256']}
          for i,t in enumerate(winners)]
    manifest={'protocol':PROTOCOL,'status':'running','args':vars(args),'source_sha256':hashes,
              'data_sha256':old['data_sha256'],'runtime':runtime_metadata(),'samples':len(y),
              'parent_run':str(parent),'candidate_metrics_sha256':hashlib.sha256((parent/'candidate_metrics.csv').read_bytes()).hexdigest(),
              'refit_plan':plan,'total_planned_epochs':sum(t['epochs'] for t in plan),
              'selection':'fixed_from_parent_inner_dev_only','test_evaluations_per_fold':1,
              'gamma':.933,'threshold':args.threshold,'refit_data':'all_outer_train_no_outer_test',
              'hard_stop':'one_refit_per_fold_no_new_configs_or_epoch_selection_after_refit',
              'claim_scope':'exploratory_protocol_alignment_on_previously_evaluated_dataset'}
    write_json(root/'manifest.json',manifest)
    full_splits=[]
    for i,(tr,dv,te) in enumerate(splits):
        for actual,k in zip((tr,dv,te),('train','dev','test')):np.testing.assert_array_equal(actual,parent_splits[i][k])
        outer=np.sort(np.concatenate([tr,dv]));assert not set(outer)&set(te)
        full_splits.append({'fold':i+1,'train':outer.tolist(),'test':te.tolist(),
                            'selection_train':tr.tolist(),'selection_dev':dv.tolist()})
    write_json(root/'splits.json',full_splits)
    start=time.time(); epochs=[]; tests=[]
    for fold,(split,choice) in enumerate(zip(full_splits,plan),1):
        outer=np.array(split['train']);te=np.array(split['test'])
        scalers=fit_scalers(raw,outer);tv=transform(raw,outer,scalers,'cuda')
        yt=y[outer].to('cuda');frequency=yt.mean(0)
        seed_all(args.seed+fold-1)
        model=VAMS([v.shape[1] for v in raw],y.shape[1],args).to('cuda')
        assert state_hash(model)==choice['initial_state_sha256']
        if choice['candidate']=='sgd': optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9,weight_decay=0)
        else: optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.933)
        print(f'Refit fold {fold}/5: {len(outer)} training rows, {choice["candidate"]}, fixed {choice["epochs"]} epochs',flush=True)
        for epoch in range(1,choice['epochs']+1):
            model.eval()
            with torch.no_grad():
                emb=model.encode(tv);neighbors=knn_indices(emb,emb,args.k,exclude_self=True)
            model.train();total=0.
            for batch,s in enumerate(range(0,len(outer),args.batch_size)):
                optimizer.zero_grad(set_to_none=True);ref=model.encode(tv)
                pred=model.predict_batch(ref[s:s+args.batch_size],ref,neighbors,batch,args)
                loss=label_frequency_loss(pred,yt[s:s+args.batch_size],frequency,args.beta)
                if not torch.isfinite(loss):raise FloatingPointError('重拟合损失非有限；停止，不临时修改配置')
                loss.backward();optimizer.step();total+=float(loss.detach())*len(pred)
            epochs.append({'fold':fold,'epoch':epoch,'candidate':choice['candidate'],
                           'lr':optimizer.param_groups[0]['lr'],'train_loss':total/len(outer)})
            scheduler.step();save_csv(root/'epoch_metrics.csv',epochs)
            if epoch==1 or epoch%10==0:print(f'  refit epoch {epoch}: train_loss={total/len(outer):.4f}',flush=True)
        torch.save({'protocol':PROTOCOL,'state_dict':model.state_dict(),'epoch':choice['epochs'],
                    'candidate':choice['candidate'],'selection_dev_LRAP':choice['selection_dev_LRAP'],
                    'initial_state_sha256':choice['initial_state_sha256'],
                    'scalers':scalers,'frequency':frequency.cpu(),'split':{'train':outer,'test':te},
                    'selection_split':{'train':np.array(split['selection_train']),'dev':np.array(split['selection_dev'])},
                    'args':vars(args),'input_dims':[v.shape[1] for v in raw],'labels':y.shape[1]},
                   root/'checkpoints'/f'fold_{fold}.pth')
        ts=predict(model,tv,transform(raw,te,scalers,'cuda'),args)
        tm=metrics(y[te],ts,args.threshold)
        tests.append({'fold':fold,'candidate':choice['candidate'],'fixed_epochs':choice['epochs'],
                      'train_rows':len(outer),'selection_dev_LRAP':choice['selection_dev_LRAP'],**tm})
        np.savez_compressed(root/f'fold_{fold}_predictions.npz',sample_ids=te,labels=y[te].numpy(),predictions=ts.cpu().numpy())
        save_csv(root/'fold_test_metrics.csv',tests)
        print(f'Refit fold {fold}: final test LRAP={tm["LRAP"]:.6f}',flush=True)
        del model,optimizer,scheduler
    summary={'status':'complete','protocol':PROTOCOL,'folds':5,'total_epochs':len(epochs),
             'seconds':time.time()-start,'mean':{k:float(np.mean([t[k] for t in tests])) for k in tm},
             'std':{k:float(np.std([t[k] for t in tests],ddof=1)) for k in tm}}
    write_json(root/'summary.json',summary);manifest['status']='complete';write_json(root/'manifest.json',manifest)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

"""真实 Emotions 数据的一轮训练/验证接线检查，不读取测试折结果，不用于效果结论。"""
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def main():
    root=Path('formula_alignment_v4')
    source=root/'source'; source.mkdir(parents=True,exist_ok=True)
    names=['demo.py','strict_training.py','paper_gn.py','paper_graph.py','GNBlock_model.py',
           'conv_layer.py','build_graph.py','MLP.py','util.py','verify_paper_alignment.py']
    hashes={}
    for name in names:
        data=Path(name).read_bytes()
        (source/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    sys.path.insert(0,str(source.resolve()))
    import strict_training as st
    import demo
    assert Path(st.__file__).resolve().parent==source.resolve()
    config=SimpleNamespace(dataset='emotions.mat',data_root='./data',seed=2020,k_fold=5,
        dev_fraction=.2,nhidden=128,target_dim=128,dropout=.4,k=5,batch_size=64,
        conv_layers=3,lr=.001,beta=1.,threshold=.5,weight_decay=0.,norm_regularizer=0.)
    demo.args=config
    views,labels=demo.data_loader('./data/emotions.mat')
    labels=torch.as_tensor(labels).float()
    if labels.shape[0]!=len(views[0]): labels=labels.T
    tr,dv,_=st.make_splits(len(labels),5,2020,.2)[0]
    # 与上一轮固定划分逐项核对，仅使用 train/dev 行进行计算。
    old=json.loads(Path('results_corrected_5fold_fold1_lr0001_80epochs/splits.json').read_text(encoding='utf-8'))[0]
    np.testing.assert_array_equal(tr,old['train']); np.testing.assert_array_equal(dv,old['dev'])
    device=torch.device('cuda')
    torch.set_num_threads(1); st.seed_all(2020)
    scalers=st.fit_scalers(views,tr)
    train=st.transform(views,tr,scalers,device)
    dev=st.transform(views,dv,scalers,device)
    y=labels[tr].to(device); frequency=y.mean(0)
    model=st.VAMS([v.shape[1] for v in views],labels.shape[1],config).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=0.)
    start=time.perf_counter()
    model.eval()
    with torch.no_grad():
        reference=model.encode(train)
        neighbors=st.knn_indices(reference,reference,5,True)
    model.train(); losses=[]
    for batch,begin in enumerate(range(0,len(tr),64)):
        optimizer.zero_grad(set_to_none=True)
        reference=model.encode(train)
        scores=model.predict_batch(reference[begin:begin+64],reference,neighbors,batch,config)
        loss=st.label_frequency_loss(scores,y[begin:begin+64],frequency,1.)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step(); losses.append(float(loss.detach()))
    scores=st.predict(model,train,dev,config)
    assert scores.shape==(len(dv),labels.shape[1]) and torch.isfinite(scores).all()
    # 内存中保存/载入参数后重放，检查新架构的可恢复性。
    buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
    replay=st.VAMS([v.shape[1] for v in views],labels.shape[1],config).to(device)
    replay.load_state_dict(torch.load(buffer,map_location=device,weights_only=True))
    torch.testing.assert_close(st.predict(replay,train,dev,config),scores)
    report={'status':'PASS','scope':'one_epoch_train_dev_wiring_only','test_evaluations':0,
        'train_rows':len(tr),'dev_rows':len(dv),'batches':len(losses),'losses':losses,
        'scores_min':float(scores.min()),'scores_max':float(scores.max()),
        'checkpoint_replay':'PASS','seconds':time.perf_counter()-start,'args':vars(config)}
    (root/'dev_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest={'protocol':st.PROTOCOL,'source_sha256':hashes,
        'data_sha256':hashlib.sha256(Path('data/emotions.mat').read_bytes()).hexdigest(),
        'paper_sha256':hashlib.sha256(Path('../paper.pdf').read_bytes()).hexdigest()}
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()

"""重复同一初始训练轮，检查 CUDA 训练轨迹敏感性；不用测试集。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deterministic', action='store_true')
    cli = p.parse_args()
    root = Path('results_scene_v4_5fold_80epochs')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    sys.path.insert(0,str((root/'source').resolve()))
    import strict_training as st
    import demo
    args = SimpleNamespace(**manifest['args'])
    demo.args=args
    views,y=demo.data_loader('data/scene.mat')
    views=[v.float() for v in views]; y=torch.as_tensor(y).T.float()
    cp=torch.load(root/'checkpoints/fold_1.pth',weights_only=False)
    tr,dv=cp['split']['train'],cp['split']['dev']
    tv=st.transform(views,tr,cp['scalers'],'cuda')
    vv=st.transform(views,dv,cp['scalers'],'cuda')
    yt=y[tr].to('cuda');frequency=yt.mean(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(cli.deterministic)
    demo.setup_seed(args.seed)
    records=[]
    for repeat in range(2):
        st.seed_all(args.seed)
        model=st.VAMS(cp['input_dims'],cp['labels'],args).to('cuda')
        h=hashlib.sha256()
        for v in model.state_dict().values(): h.update(v.detach().cpu().numpy().tobytes())
        opt=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9)
        model.eval()
        with torch.no_grad():
            emb=model.encode(tv);neighbors=st.knn_indices(emb,emb,args.k,exclude_self=True)
        model.train(); losses=[]
        for b,s in enumerate(range(0,len(tr),args.batch_size)):
            opt.zero_grad(set_to_none=True)
            ref=model.encode(tv)
            scores=model.predict_batch(ref[s:s+args.batch_size],ref,neighbors,b,args)
            loss=st.label_frequency_loss(scores,yt[s:s+args.batch_size],frequency,args.beta)
            loss.backward();opt.step()
            losses.append(float(loss.detach()))
        vs=st.predict(model,tv,vv,args).cpu().numpy()
        if repeat==0: first_scores=vs
        records.append({'repeat':repeat,'initial_state_sha256':h.hexdigest(),
                        'batch_losses':losses,'dev_LRAP':st.metrics(y[dv],torch.from_numpy(vs),.5)['LRAP']})
        del model
    report={'deterministic':cli.deterministic,'test_evaluations':0,'repeats':records,
            'max_dev_probability_change_between_repeats':float(np.abs(first_scores-vs).max())}
    suffix='deterministic' if cli.deterministic else 'standard'
    Path(f'diagnosis_scene_gap/numerics_{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

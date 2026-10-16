"""仅改变标签节点的遍历编号，固定身份向量和权重，检查预测的置换等变性。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--implementation', choices=['snapshot','workspace'], default='snapshot')
    p.add_argument('--require-invariant', action='store_true')
    cli=p.parse_args()
    root=Path(cli.run).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest['source_sha256'].items():
        assert hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==h,n
    sys.path.insert(0,str(root/'source' if cli.implementation=='snapshot' else Path(__file__).resolve().parent))
    from strict_training import VAMS,transform,knn_indices
    import demo
    args=SimpleNamespace(**manifest['args']); demo.args=args
    torch.set_num_threads(1)
    views,_=demo.data_loader(str(Path(args.data_root)/args.dataset))
    cp=torch.load(root/'checkpoints/fold_1.pth',map_location='cpu',weights_only=False)
    model=VAMS(cp['input_dims'],cp['labels'],args); model.load_state_dict(cp['state_dict']); model.eval()
    with torch.no_grad():
        reference=model.encode(transform(views,cp['split']['train'],cp['scalers'],'cpu'))
        # 完整外层训练重拟合已吸收旧验证行，只用这批训练子集做结构回归。
        query_ids=cp['split']['dev'] if 'dev' in cp['split'] else cp['selection_split']['dev']
        query_role='inner_dev' if 'dev' in cp['split'] else 'training_subset_former_inner_dev'
        query=model.encode(transform(views,query_ids,cp['scalers'],'cpu'))
        neighbors=knn_indices(reference,query,args.k)
        def predict():
            return torch.cat([model.predict_batch(query[s:s+args.batch_size],reference,neighbors,b,args)
                for b,s in enumerate(range(0,len(query),args.batch_size))])
        baseline=predict(); identity=model.label_identity.clone(); records=[]
        for perm in [list(reversed(range(model.labels))),list(range(1,model.labels))+[0]]:
            model.label_identity=identity[perm]
            # 节点位置和 identity 同时移动；输出逆置换还原真实标签语义。
            changed=predict()[:,np.argsort(perm)]
            records.append({'permutation':perm,'max_abs_change':float((changed-baseline).abs().max()),
                            'mean_abs_change':float((changed-baseline).abs().mean()),
                            'changed_threshold_positions':int(((changed>=args.threshold)!=(baseline>=args.threshold)).sum()),
                            'invariance_at_1e_5':bool(torch.allclose(changed,baseline,atol=1e-5,rtol=1e-5))})
        model.label_identity=identity
        torch.testing.assert_close(predict(),baseline)
    report={'run':root.name,'source_verified':True,'implementation':cli.implementation,
            'no_training':True,'test_evaluations':0,'query_role':query_role,'label_permutations':records}
    out=Path('diagnosis_paper_run');out.mkdir(exist_ok=True)
    suffix='' if cli.implementation=='snapshot' else '_workspace'
    (out/('label_order_'+root.name+suffix+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if cli.require_invariant:
        assert all(r['invariance_at_1e_5'] for r in records), '标签身份不变，编号变化不应改变语义预测'

if __name__=='__main__': main()

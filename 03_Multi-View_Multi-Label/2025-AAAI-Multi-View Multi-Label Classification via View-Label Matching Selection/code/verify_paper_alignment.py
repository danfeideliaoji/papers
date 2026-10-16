"""公式对齐回归检查：独立逐邻接点计算式 (4)-(9)，不以模型分数判断正确性。"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    cli.add_argument('--output_dir', default='formula_alignment_v4')
    args = cli.parse_args()
    import paper_gn as gn
    # 旧实现缺少独立邻居聚合和边的 rho/phi，两条结构断言必须先失败。
    node = gn.NodeModel(128, 128, 128, 128, 128)
    assert hasattr(node, 'node_mlp_ins_neighbor'), '式 (4) 的邻居聚合仍与视图聚合共用一个网络'
    edge = gn.EdgeModel(128, 128, 128, 128, 128)
    assert hasattr(edge, 'edge_rho_ins'), '式 (7) 缺少独立的端点聚合 MLP'
    from paper_graph import Graph_Generator
    from strict_training import VAMS, label_frequency_loss, make_splits, fit_scalers, transform, knn_indices
    torch.set_num_threads(1)
    torch.manual_seed(2020)
    device = torch.device(args.device)
    # 保留前一版本的独立信息边界检查。
    splits = make_splits(593,5,2020,.2)
    assert sorted(i for _,_,te in splits for i in te)==list(range(593))
    for tr,dv,te in splits:
        assert not(set(tr)&set(dv) or set(tr)&set(te) or set(dv)&set(te))
    raw_scaler=[torch.tensor([[2.],[4.],[100.]])]
    scalers=fit_scalers(raw_scaler,[0,1]); raw_scaler[0][2]=-10000
    for original,new in zip(scalers[0],fit_scalers(raw_scaler,[0,1])[0]):
        torch.testing.assert_close(original,new)
    assert transform(raw_scaler,[2],scalers,device)[0].item()<0
    duplicate=torch.ones(9,2,128)
    nn_ids=knn_indices(duplicate,duplicate,3,True)
    assert all(i not in nn_ids[v,i].tolist() for i in range(9) for v in range(2))
    assert knn_indices(duplicate,torch.ones(4,2,128),3).max()<9
    calculated=label_frequency_loss(torch.full((2,1),.5),torch.tensor([[1.],[0.]]),torch.tensor([.25]),1.)
    torch.testing.assert_close(calculated,torch.tensor(1.1787057),atol=1e-6,rtol=1e-6)
    config = SimpleNamespace(nhidden=128, target_dim=128, dropout=.4, k=2,
                             batch_size=4, conv_layers=3, beta=1., knn_space='raw')
    generator = Graph_Generator()
    checks = []
    for views in [1, 2, 3]:
        for batch in [1, 3]:
            labels, k, h = 4, 2, 128
            sample = torch.randn(batch, views, h, device=device, requires_grad=True)
            bank = torch.randn(9, views, h, device=device, requires_grad=True)
            identity = torch.randn(labels, h, device=device, requires_grad=True)
            ids = torch.arange(batch*k).reshape(1, batch, k).expand(views, -1, -1)
            placeholder = torch.zeros(views, batch, labels, device=device)
            gi, gl = generator(sample, identity, placeholder, config, ids, bank, 0)
            gc = gn._Model().generate_cross_graph(gl, torch.tensor(gi.kwargs), k)
            stride = views*(1+k)
            assert gi.edge_index.shape[1] == batch*(views*(views-1)+views*k)
            assert gl.edge_index.shape[1] == batch*labels*(labels-1)
            assert gc.edge_index.shape[1] == batch*views*labels
            assert not torch.any(gi.edge_index[0] == gi.edge_index[1])
            # 有序拼接与接收/发送角色对应，每个方向恰好一条消息。
            for graph in [gi, gl]:
                torch.testing.assert_close(graph.edge_attr,
                    torch.cat([graph.x[graph.edge_index[0]], graph.x[graph.edge_index[1]]], -1))
                edges = [tuple(e) for e in graph.edge_index.T.tolist()]
                assert len(set(edges)) == len(edges)
            torch.testing.assert_close(gi.x.reshape(batch, stride, h)[:, views:],
                torch.stack([bank[ids[v], v] for v in range(views)], 1).reshape(batch, views*k, h))
            si = (gi.x, gi.edge_index, torch.randn(gi.edge_index.shape[1], h, device=device))
            sl = (gl.x, gl.edge_index, torch.randn(gl.edge_index.shape[1], h, device=device))
            sc = (gc.edge_index, torch.randn(gc.edge_index.shape[1], h, device=device))
            node = gn.NodeModel(h,h,h,h,h).to(device)
            node.num_views, node.num_neighbors = views, k
            ni, nl, _ = node(si, sl, sc)
            # 独立 Python 邻接遍历，检查三种视图消息的分组与归一化。
            for b in range(batch):
                for v in range(views):
                    n = b*stride+v
                    other, neighbor, match = [], [], []
                    for e, (s,t) in enumerate(si[1].T.tolist()):
                        if n != s: continue
                        source = t
                        msg = torch.cat([si[2][e], si[0][source]])
                        if source % stride < views: other.append(node.node_mlp_ins_inner(msg))
                        else: neighbor.append(node.node_mlp_ins_neighbor(msg))
                    for e, (s,t) in enumerate(sc[0].T.tolist()):
                        if s == n: match.append(node.node_mlp_ins_inter(torch.cat([sc[1][e], sl[0][t]])))
                    mean = lambda xs: torch.stack(xs).mean(0) if xs else si[0].new_zeros(h)
                    # 式 (4) 标签通道没有 1/Q：按字面采用求和。
                    expected = node.node_mlp_ins(torch.cat([si[0][n], mean(other), mean(neighbor), torch.stack(match).sum(0)]))
                    torch.testing.assert_close(ni[0][n], expected, atol=3e-5, rtol=3e-5)
            for n in range(batch*labels):
                other, match = [], []
                for e, (s,t) in enumerate(sl[1].T.tolist()):
                    if n == s:
                        source = t
                        other.append(node.node_mlp_label_inner(torch.cat([sl[2][e], sl[0][source]])))
                for e,(s,t) in enumerate(sc[0].T.tolist()):
                    if t == n: match.append(node.node_mlp_label_inter(torch.cat([sc[1][e], si[0][s]])))
                expected = node.node_mlp_label(torch.cat([sl[0][n], torch.stack(other).mean(0), torch.stack(match).mean(0)]))
                torch.testing.assert_close(nl[0][n], expected, atol=3e-5, rtol=3e-5)
            # 式 (7)-(9)：rho(端点拼接) -> phi(旧边与 rho 的拼接)。
            edge = gn.EdgeModel(h,h,h,h,h).to(device)
            ei, el, ec = edge(ni,nl,sc)
            for state, updated, prefix, left, right in [(ni,ei,'ins',ni[0],ni[0]), (nl,el,'label',nl[0],nl[0])]:
                rho, phi = getattr(edge,'edge_rho_'+prefix), getattr(edge,'edge_mlp_'+prefix)
                expected = phi(torch.cat([state[2], rho(torch.cat([left[state[1][0]],right[state[1][1]]],-1))],-1))
                torch.testing.assert_close(updated[2],expected)
            expected = edge.edge_mlp_cross(torch.cat([sc[1], edge.edge_rho_cross(torch.cat([ni[0][sc[0][0]],nl[0][sc[0][1]]],-1))],-1))
            torch.testing.assert_close(ec[1],expected)
            checks.append({'views':views,'batch':batch,'formula_oracle':'PASS'})
    model = VAMS([5,7,3],4,config).to(device)
    raw = [torch.randn(9,d,device=device) for d in [5,7,3]]
    reference = model.encode(raw)
    ids = torch.tensor([[[1,2],[2,3],[3,4]]] * 3)
    sequence=[]
    handles=[]
    for i,layer in enumerate(model.graph.layers):
        handles.append(layer.node_model.register_forward_hook(lambda m,a,o,i=i: sequence.append((i,'node'))))
        handles.append(layer.edge_model.register_forward_hook(lambda m,a,o,i=i: sequence.append((i,'edge'))))
    scores = model.predict_batch(reference[:3],reference,ids,0,config)
    for handle in handles: handle.remove()
    assert sequence==[(i,kind) for i in range(3) for kind in ['node','edge']]
    assert scores.shape == (3,4) and torch.isfinite(scores).all()
    target = torch.tensor([[1,0,0,1],[0,1,1,0],[1,0,1,0]],device=device,dtype=torch.float)
    frequency = target.mean(0)
    loss = label_frequency_loss(scores,target,frequency,1.)
    expected = -(torch.exp(target*(1-frequency)+(1-target)*frequency)*(target*scores.log()+(1-target)*(1-scores).log())).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    missing = [n for n,p in model.named_parameters() if p.grad is None]
    # 最终图内边没有后续节点读取；这是式 (10) 只解码匹配边的自然终点。
    allowed = 'graph.layers.2.edge_model.'
    assert all(n.startswith(allowed) and '_cross.' not in n for n in missing), missing
    assert len(missing)==32, missing
    for n,p in model.named_parameters():
        if n not in missing: assert torch.isfinite(p.grad).all()
        if 'layers.2.node_model' in n: assert p.grad is not None and p.grad.abs().sum() > 0, n
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in projection.parameters()) for projection in model.input_projections)
    before=[p.detach().clone() for p in model.input_projections.parameters()]
    # 非首批、尾批也必须正确索引邻居；逐样本前向不能与其他样本串图。
    model.eval()
    reference = model.encode(raw)
    all_ids = torch.tensor([[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]] * 3)
    config.batch_size = 4
    tail = model.predict_batch(reference[4:6],reference,all_ids,1,config)
    config.batch_size = 1
    singleton = torch.cat([model.predict_batch(reference[i:i+1],reference,all_ids,i,config) for i in [4,5]])
    torch.testing.assert_close(tail,singleton,atol=3e-5,rtol=3e-5)
    config.batch_size = 4
    # 真实标签不影响图结构/预测：占位值即使改变，结果仍相同。
    embedded_labels=model.label_projection(model.label_identity)[None]
    prediction0,_=model.graph(reference[4:6],embedded_labels,torch.zeros(2,3,4,device=device),config,all_ids[None],reference[None],1)
    prediction1,_=model.graph(reference[4:6],embedded_labels,torch.ones(2,3,4,device=device),config,all_ids[None],reference[None],1)
    torch.testing.assert_close(prediction0,prediction1)
    # 核心行为回归：节点身份不变，编号改变并逆置换输出后，语义预测必须一致。
    baseline_identity=model.label_identity.clone()
    reference_scores=model.predict_batch(reference[4:6],reference,all_ids,1,config)
    for perm in [[3,2,1,0],[1,2,3,0]]:
        model.label_identity=baseline_identity[perm]
        reordered=model.predict_batch(reference[4:6],reference,all_ids,1,config)
        inverse=torch.argsort(torch.tensor(perm,device=device))
        torch.testing.assert_close(reordered[:,inverse],reference_scores,atol=1e-5,rtol=1e-5)
    model.label_identity=baseline_identity
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    opt.step()
    assert any(not torch.equal(old,new) for old,new in zip(before,model.input_projections.parameters()))
    result = {'status':'PASS','device':str(device),'checks':checks,'task_loss':float(loss.detach()),
              'last_node_nonzero_gradient':True,'terminal_unused_edge_parameters':missing,
              'split_scaler_knn_information_boundary':'PASS','batch_isolation_and_label_placeholder':'PASS',
              'algorithm_1_node_then_edge':'PASS',
              'label_order_invariance':'PASS',
              'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                  ['paper_gn.py','paper_graph.py','conv_layer.py','strict_training.py','MLP.py','verify_paper_alignment.py']}}
    root=Path(args.output_dir); root.mkdir(exist_ok=True)
    (root / ('verification_'+args.device+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()

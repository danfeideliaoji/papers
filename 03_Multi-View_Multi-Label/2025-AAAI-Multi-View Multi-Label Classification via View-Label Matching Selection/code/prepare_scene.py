"""核对 Mulan 官方 Scene ARFF，按作者现有 98/196 规则生成 MAT 输入。

仅转换存储格式与视图分组，不拟合预处理；训练时仍只在各折训练行归一化。
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import arff, savemat

def main():
    source=Path('data/scene_source')
    labels=[element.attrib['name'] for element in ET.parse(source/'scene.xml').getroot()]
    def load(name):
        rows,meta=arff.loadarff(source/name)
        names=meta.names()
        assert names[-len(labels):]==labels
        x=np.column_stack([rows[n].astype(np.float64) for n in names[:-len(labels)]])
        y=np.column_stack([rows[n].astype(np.float64) for n in labels])
        assert x.shape[1]==294 and y.shape[1]==6
        assert np.isfinite(x).all() and np.all((y==0)|(y==1))
        return x,y,names
    xt,yt,names=load('scene-train.arff');xe,ye,_=load('scene-test.arff')
    x,y,_=load('scene.arff')
    assert len(xt)==1211 and len(xe)==1196 and len(x)==2407
    np.testing.assert_array_equal(x,np.concatenate([xt,xe]));np.testing.assert_array_equal(y,np.concatenate([yt,ye]))
    data=np.empty((1,2),dtype=object);data[0,0]=x[:,:98].T;data[0,1]=x[:,98:].T
    output=Path('data/scene.mat')
    if output.exists():raise FileExistsError('拒绝覆盖已有 Scene 数据')
    savemat(output,{'data':data,'target':y.T},do_compression=True)
    record={'dataset':'Scene','source_url':'https://sourceforge.net/projects/mulan/files/datasets/scene.rar/download',
        'catalog':'https://mulan.sourceforge.net/datasets-mlc.html','rows':2407,'official_train_rows':1211,
        'official_test_rows':1196,'labels':labels,'view_dimensions':[98,196],
        'feature_grouping':'first_98_and_remaining_196_from_author_demo_scene_loader',
        'paper_view_definition':'not_explicitly_specified','preprocessing_at_conversion':'none',
        'union_matches_train_then_test':True,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
        'mat_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (source/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()

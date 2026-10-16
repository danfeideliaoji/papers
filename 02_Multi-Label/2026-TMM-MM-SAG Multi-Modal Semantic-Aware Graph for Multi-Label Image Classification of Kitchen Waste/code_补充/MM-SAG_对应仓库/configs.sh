cd LMPT
# pip install --upgrade torch==1.13 -i https://pypi.mirrors.ustc.edu.cn/simple
### cosine-annealing-warmup==2.0
cd pytorch-cosine-annealing-with-warmup
python setup.py install
cd ..
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
python setup_clip.py install
pip install mmcv==1.7.0 -i https://pypi.mirrors.ustc.edu.cn/simple
pip install transformers==4.20.0 -i https://pypi.mirrors.ustc.edu.cn/simple
pip install torchnet -i https://pypi.mirrors.ustc.edu.cn/simple
pip install yacs -i https://pypi.mirrors.ustc.edu.cn/simple
pip install cupy-cuda116 -i https://pypi.mirrors.ustc.edu.cn/simple
pip install tqdm -i https://pypi.mirrors.ustc.edu.cn/simple
pip install timm -i https://pypi.mirrors.ustc.edu.cn/simple
pip install tensorboard -i https://pypi.mirrors.ustc.edu.cn/simple
pip install prefetch-generator==1.0.1 -i https://pypi.mirrors.ustc.edu.cn/simple
pip install neptune-client -i https://pypi.mirrors.ustc.edu.cn/simple
pip install scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple
# pip install seaborn 
pip install grad-cam==1.4.8
#cd LMPT
###mmcv
#python lmpt/train.py

#AttributeError: 'tuple' object has no attribute 'norm':解决方案:clip 文件夹错误




##############clip==1.0安装
 





# clip==1.0
# sklearn==0.0.post1
# google-auth-oauthlib==0.4.6




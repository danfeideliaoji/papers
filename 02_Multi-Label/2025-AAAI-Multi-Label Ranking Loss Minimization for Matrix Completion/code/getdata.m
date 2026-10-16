function [left,DATA,right] = getdata(mode,dataId)
if mode=="MLL"
    datasets = {'Arts','Business','Computers','Education','Health','Recreation','Society'};
    data_path = 'data_MLL/';
    data = load(strcat(data_path,datasets{dataId},'.mat'));
    [left, ~] = qr([data.train_data; data.test_data], 0);
%     left = [data.train_data; data.test_data];
    DATA = [data.train_target; data.test_target];
    right = eye(size(DATA,2));
end
if mode=="movie"
    dt = load('data_movie/movielens_data.mat');
    left = dt.X;
    DATA = dt.E_org;
    right = dt.Y;
end
if mode=="DTI"
    datasets = {'Enzymes','GPCRs'};
    data_path = 'data_DTI/';
    dt = load(strcat(data_path,datasets{dataId},'.mat'));
    left = fPCA(dt.LEFT,0.9);
    DATA = dt.DATA;
    right = fPCA(dt.RIGHT,0.9);
end
end
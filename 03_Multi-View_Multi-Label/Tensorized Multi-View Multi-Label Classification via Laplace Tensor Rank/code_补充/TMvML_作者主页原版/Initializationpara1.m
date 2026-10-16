function [Ktrain]=Initializationpara(X_set, kernel_type, kernel_para)

num_views=length(X_set);
Ktrain=cell(num_views,1);

options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 10;  % nearest neighbor
options.WeightMode = 'HeatKernel';
options.t = 1.0;
for i = 1:num_views
    Ktrain{i} = kernelmatrix(kernel_type,X_set{i}',X_set{i}',kernel_para); % n by n, kernel matrix
    
end
end
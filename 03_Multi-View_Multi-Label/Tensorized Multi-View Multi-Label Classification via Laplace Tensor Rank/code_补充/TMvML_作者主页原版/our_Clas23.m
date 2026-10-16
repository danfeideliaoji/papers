function [output_args,Theta_vec] = our_Clas23( X,Y,train_index,para )
%DSML_MODEL Summary of this function goes here
%   Detailed explanation goes here
% input: X m*1 cell, each cell is a matrix of size n*dv
% Y: label indicator of l labeled instance l+u
% train_index: index of train samples



nV=length(X); %m=2
Train_index=zeros(size(Y));
Train_index(train_index,:)=1; %将Train_index第train_index行全变成1
alpha=para.alpha;
delta=para.beta;
gamma=para.gama;

[N,q]=size(Y); %取行列n，q
t=para.dim;  %k的值

% end
    for i = 1:nV
        X{i} = normalization(X{i}, 'l2', 1);
    end
test_index = setdiff(1:N, train_index);
%Y=sparse(Y);
% Y1=Y;
% Y=Y(train_index,:);
%% ============================ Initialization ============================
for k=1:nV
    Z{k} = zeros(N,t); 
    C{k} = zeros(t,q);
    J{k} = zeros(t,q);
    W{k} = zeros(t,q);
    A{k} = zeros(t,size(X{k},2));
    E{k} = zeros(N,size(X{k},2)); %E{2} = zeros(size(X{k},1),N);
    B{k} = zeros(N,size(X{k},2)); %Y{2} = zeros(size(X{k},1),N);
end
S=eye(N,N);
S(test_index,test_index)=0;

c = zeros(t*q*nV,1);
j = zeros(t*q*nV,1);
sX = [t, q, nV];%张良的形状

Isconverg = 0;epson = 1e-5;
iter = 1;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;
Convergence=zeros(100,1);
converge_Z=[];
converge_W_G=[];


%% ================================ Upadate ===============================
while(Isconverg == 0)
%% =========================== Upadate E^k, B^k ===========================
      temp_E =[];
      for k =1:nV           
           temp_E=[temp_E,X{k}-Z{k}*A{k}+B{k}/mu];
       end
       temp_E=temp_E';
       [Econcat] = solve_l1l2(temp_E,alpha/mu);
       ro_b =0;
       E{1} =  Econcat(1:size(X{1},2),:)';
       B{1} = B{1} + mu*(X{1}-Z{1}*A{1}-E{1});
       ro_end = size(X{1},2);
       for i=2:nV
           ro_b = ro_b + size(X{i-1},2);
           ro_end = ro_end + size(X{i},2);
           E{i} =  Econcat(ro_b+1:ro_end,:)';
           B{i} = B{i} + mu*(X{i}-Z{i}*A{i}-E{i});
       end
    %% ============================== Upadate W ===============================
      for k =1:nV
          tmp2 = -C{k}+2*Z{k}'*S*Y +rho*J{k} ;
           %W{k}=inv(Z{k}'*Z{k}+rho*eye(t,t))*tmp2;
          W{k}=lsqminnorm(2*Z{k}'*S*Z{k}+(rho)*eye(t,t),tmp2);
      end
      %% ============================== Upadate A{v} ===============================
   G={};
for i = 1 :nV
    G{i}=Z{i}'*(B{i}+mu*X{i}-mu*E{i});
    [Au,ss,Av] = svd(G{i},'econ');
    A{i}=Au*Av';
end

%% ============================== Upadate Z^k =============================
     clear i l
       
      for k =1:nV
          tmp = 2*Y(train_index,:)*W{k}' + B{k}(train_index,:)*A{k}' +  mu*X{k}(train_index,:)*A{k}'- mu*E{k}(train_index,:)*A{k}' ;
          S_b_len = size(W{k}*W{k}');
          Z_down =2*W{k}*W{k}'+mu*eye(S_b_len(1)); 
          Z{k}(train_index,:)=lsqminnorm(Z_down',tmp')';
      end
      for k =1:nV
          tmp = B{k}(test_index,:)*A{k}' +  mu*X{k}(test_index,:)*A{k}'- mu*E{k}(test_index,:)*A{k}' ;

          S_b_len = size(W{k}*W{k}');
          Z_down =mu*eye(S_b_len(1));
          Z{k}(test_index,:)=tmp/(Z_down);

      end
      clear k 
%% ============================= Upadate J^k ==============================

                C_tensor = cat(3, C{:,:});
                W_tensor = cat(3, W{:,:});
                c = C_tensor(:);
                w = W_tensor(:);
                J_tensor = solve_G2(W_tensor + 1/rho*C_tensor,rho,sX,delta,gamma);
                %J_tensor = solve_G1(W_tensor + 1/rho*C_tensor,rho,sX,delta,gamma);
                j = J_tensor(:);
                %TNN
                % [j,objV] = wshrinkObj(W_tensor + 1/rho*C_tensor,1/rho,sX,0,3);
                % J_tensor=reshape(j, sX);
%% ============================== Upadate C ===============================
        c = c + rho*(w - j);
        C_tensor = reshape(c, sX);
    for k=1:nV
        C{k} = C_tensor(:,:,k);
    end

%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_W_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}-Z{k}*A{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-Z{k}*A{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        J{k} = J_tensor(:,:,k);
        C_tensor = reshape(c, sX);
        C{k} = C_tensor(:,:,k);
        if (norm(W{k}-J{k},inf)>epson)
            history.norm_W_G = norm(W{k}-J{k},inf);
            Isconverg = 0;
            max_W_G=max(max_W_G, history.norm_W_G);
        end
        
    end
    converge_Z=[converge_Z max_Z];
    converge_W_G=[converge_W_G max_W_G];
    for vv=1:nV
        Convergence(iter,1)=Convergence(iter,1)+norm(S*((Z{vv}*W{vv})-Y),'fro')^2;
    end
    if iter>1
        if abs(Convergence(iter,1)-Convergence(iter-1,1))>(1e-4)*Convergence(iter-1,1)%比前一次迭代更新少于1e-5时停止
            
            Isconverg  = 0;
        end
    end
    if (iter>50)
        Isconverg  = 1;
    end
    if (Isconverg  == 1)
 %       fprintf('\n ');
 %       disp(strcat('end at the ',31,num2str(iter),'th iteration'));
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end
%% output求得y帽
W_output=zeros(size(Y));
W_output=W_output(test_index,:);
for vv=1:nV
    W_output=W_output+Z{vv}(test_index,:)*W{vv};
end
output_args=(W_output/nV);
output_args=abs(output_args);

end


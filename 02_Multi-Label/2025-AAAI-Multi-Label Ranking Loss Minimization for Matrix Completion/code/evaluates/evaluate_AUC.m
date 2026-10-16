function auc = evaluate_AUC(predict,target,posTest)
test_idx = posTest==1;
pred = predict(test_idx);
targ = target(test_idx);

[~,idx] = sort(pred, 'descend');
label = targ(idx);
tp = cumsum(label == 1);
fp = sum(label == 0);
auc = sum(tp(label == 0));
if tp == 0 | fp == 0
    disp("Warning: postive or negative labels only.");
    auc = 0;
else
	auc = auc / tp(end) / fp;
end
end
% Mean Absolute Error
% function loss = evaluate_MAE(predict,target,obvT)
% error = sum(sum(abs((predict-target).*(1-obvT))));
% base = sum(sum(1-obvT));
% loss = error/base;
% end

function loss = evaluate_MAE(predict,target,posTest)
error = sum(sum(abs((predict-target).*(posTest))));
base = sum(sum(posTest));
loss = error/base;
end
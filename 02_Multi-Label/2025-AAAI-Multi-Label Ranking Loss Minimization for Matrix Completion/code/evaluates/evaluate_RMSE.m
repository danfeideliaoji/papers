% Root Mean Square Error
% function loss = evaluate_RMSE(predict,target,obvT)
% error = norm((predict-target).*(1-obvT), 'fro');
% base = sqrt(sum(sum(1-obvT)));
% loss = error/base;
% end

function loss = evaluate_RMSE(predict,target,posTest)
error = norm((predict-target).*(posTest), 'fro');
base = sqrt(sum(sum(posTest)));
loss = error/base;
end
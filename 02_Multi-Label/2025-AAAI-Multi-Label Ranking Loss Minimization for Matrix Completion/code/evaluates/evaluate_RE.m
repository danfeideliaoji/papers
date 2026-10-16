% Relative Error
% function loss = evaluate_RE(predict,target,obvT)
% error = norm((predict-target).*(1-obvT), 'fro');
% base = norm(target.*(1-obvT), 'fro');
% loss = error/base;
% end

function loss = evaluate_RE(predict,target,posTest)
error = norm((predict-target).*(posTest), 'fro');
base = norm(target.*(posTest), 'fro');
loss = error/base;
end
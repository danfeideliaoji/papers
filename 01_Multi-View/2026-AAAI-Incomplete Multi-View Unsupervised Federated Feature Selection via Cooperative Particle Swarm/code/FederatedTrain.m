function [global_W, last_mean_NMI] = FederatedTrain(X, Y, c, V, K, client_indices, options, max_comm_rounds)
    for v = 1:V
        global_W{v} = zeros(size(X{v},1), c); 
    end
    last_mean_NMI = 0;
    
    % Check if parallel pool is running, if not, start one
    if isempty(gcp('nocreate'))
        parpool('local');
    end

    for comm = 1:max_comm_rounds
        disp(['--- Federated Communication Round: ', num2str(comm), ' ---']);
        
        local_W = cell(1, K);
        local_NMI = zeros(1, K);
        
        parfor k = 1:K
            X_local = cell(1, V);
            for v = 1:V
                X_local{v} = X{v}(:, client_indices{k});
            end
            Y_local = Y(client_indices{k});
            
            % Local training on client k
            [W_local, local_NMI(k)] = ClientLocalTrain(X_local, Y_local, c, options);
            local_W{k} = W_local;
        end
        
        % Aggregating the results from all clients
        global_W = ServerAggregate(local_W, local_NMI, V, K);
        
        % Check for convergence, stop if the change in NMI is small
        if comm > 1 && abs(mean(local_NMI) - last_mean_NMI) < 1e-3
            disp('Global convergence, stopping early.');
            break;
        end
        last_mean_NMI = mean(local_NMI);
    end
end

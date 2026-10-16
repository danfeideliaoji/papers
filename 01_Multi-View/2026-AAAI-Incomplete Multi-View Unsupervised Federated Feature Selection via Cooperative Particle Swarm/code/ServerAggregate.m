function global_W = ServerAggregate(local_W, local_NMI, V, K)
agg_weights = local_NMI / sum(local_NMI + eps);
for v = 1:V
    W_sum = 0;
    for k = 1:K
        W_sum = W_sum + agg_weights(k) * local_W{k}{v};
    end
    global_W{v} = W_sum;
end
end 
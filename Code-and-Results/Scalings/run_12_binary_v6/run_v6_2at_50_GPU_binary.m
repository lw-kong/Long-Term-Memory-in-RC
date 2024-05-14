% run_1
% Now we are recording the rmse for all the val trials, not just the
% largest one.

% run_2
% now using func_STP_train_NullDim_50 with sparse matrices

addpath '..'
addpath '../..'

%gpuDevice(4)


n_set = 120:10:320;
repeat_num = 400;
tp_order = 1:2;


plot_rand_num = 0;




tp_W = 2;
tp_bias = -3;
%tp_train_set =  tp_W * eye(length(tp_order)) +tp_bias; % one hot
tp_train_set =  tp_W * cartprod(1:2) +tp_bias;

train_r_step_length = 1e3;
noise_a = 10^(-3);
paras_error = [25, 1000, 800,0.1,0.2]; % warmup_len, predict_len, rmse_len, pers_thres
dim = 3;

rng((now*1000-floor(now*1000))*100000)
filename_save = ['save_run_v6_notp_50_GPU_tp' num2str(length(tp_order)) ...
    '_' datestr(now,30) '_0_' num2str(randi(999)) '.mat'];

if length(n_set)*repeat_num*length(tp_order) >= 100
    plot_rand_num = 0;
end

tic
result_reconError = zeros(length(n_set),repeat_num);
result_rmse_set = zeros(length(n_set),repeat_num,length(tp_order));
result_pers_set_region =  zeros(length(n_set),repeat_num,length(tp_order));
result_pers_set_new1 =  zeros(length(n_set),repeat_num,length(tp_order));
result_pers_set_new2 =  zeros(length(n_set),repeat_num,length(tp_order));
result_W_out = cell(length(n_set),repeat_num);
for n_i = 1:length(n_set)
    n = n_set(n_i);
    for repeat_i = 1:repeat_num
        reconError = NaN;
        while isnan(reconError)
            [reconError,rmse_set,pers_len_set_region,...
                pers_len_set_new1,pers_len_set_new2,W_in,W_r,W_out] = ...
                func_v6_STP_train_MulChannel_50_GPU(n,tp_train_set,...
                train_r_step_length,noise_a,paras_error,plot_rand_num);
        end
        
        result_reconError(n_i,repeat_i) = reconError;
        fprintf('attempt reconError = %f\n',reconError)
        result_rmse_set(n_i,repeat_i,:) = rmse_set;    
        fprintf('attempt max rmse = %f\n',max(rmse_set))

        result_pers_set_region(n_i,repeat_i,:) = pers_len_set_region;
        fprintf('attempt min pers region = %f\n',min(pers_len_set_region))
        result_pers_set_new1(n_i,repeat_i,:) = pers_len_set_new1;
        fprintf('attempt min pers new1 = %f\n',min(pers_len_set_new1))
        result_pers_set_new2(n_i,repeat_i,:) = pers_len_set_new2;
        fprintf('attempt min pers new2 = %f\n',min(pers_len_set_new2))

        result_W_out{n_i,repeat_i} = W_out;


        fprintf('%f is done\n',((n_i-1)*repeat_num+repeat_i)/(length(n_set)*repeat_num))
        toc
        run_time = toc;
    end
    save(filename_save,'-v7.3')
    
    fprintf('size %d is done\n\n\n',n)
    
end






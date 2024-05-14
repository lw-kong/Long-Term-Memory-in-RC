function [reconError,rmse_set,pers_len_set_region,...
    pers_len_set_new1,pers_len_set_new2,W_in,W_r,W_out] = ...
    func_v6_STP_train_MulChannel_50_GPU(n,tp_train_set,...
    train_r_step_length,noise_a,paras_error,plot_rand_num)

% chose one of the best bo training results
% different tp_set and ode_parameter_set
% warm up
% no return map validation
% control parameter is 1 dim

% v2 output more
% v3 real long-term memory testing, include
%    r random reinit & warming


dim = 3;

eig_rho = 0.39;
W_in_a = 0.91;
a = 0.64;
beta = 1 * 10^(-6.5);
%k = round(799/2e3*n);
k = round(10/2e3*n);


W_in_type = 1;

warmup_r_step_length = paras_error(1);
predict_r_step_length = paras_error(2);
rmse_len = paras_error(3);
pers_thres1 = paras_error(4);
pers_thres2 = paras_error(5);




dim_tp = size(tp_train_set,2);
dim_set = 3*ones(1,size(tp_train_set,1));
rng('shuffle');
%tic;

validate_r_step_length = 1; % actually useless
predict_r_step_cut = 0;

%% preparing training data
train_data_length = train_r_step_length + validate_r_step_length + 10;
train_data = zeros(length(tp_train_set), train_data_length,dim+dim_tp); % data that goes into reservior_training

%load('data_S16_0.mat')
load('s0_1e4_t200_0.mat')
train_select_set = randperm(size(data_all,1));
train_select_set = train_select_set(1:size(tp_train_set,1));
for tp_i = 1:size(tp_train_set,1)
    tp = tp_train_set(tp_i,:);

    start_step_i = randi(size(data_all,2));
    train_data_sample = zeros(size(data_all,2),dim);
    train_data_sample(:,:) = data_all( train_select_set(tp_i),:,:);
    train_data_sample = repmat(train_data_sample,...
        [ round((train_r_step_length+validate_r_step_length)/size(data_all,2))+5,1]);
    train_data(tp_i,:,1:dim) = train_data_sample( ...
        start_step_i:start_step_i-1+train_data_length,:);

    train_data(tp_i,:,dim+1:dim+dim_tp) = repmat(tp,train_data_length,1);    %% system sensitive
end


%% train
flag_r_train = [n k eig_rho W_in_a a beta 0 train_r_step_length validate_r_step_length...
    1 dim noise_a];
[reconError,W_in,W_r,W_out] = func_STP_train_NullDim_50_GPU_reconError(...
    train_data,tp_train_set,dim_set,flag_r_train,W_in_type,2,1);

%% recall and test
rmse_set = zeros(size(tp_train_set,1),1);
pers_len_set_new1 = zeros(size(tp_train_set,1),1);
pers_len_set_new2 = zeros(size(tp_train_set,1),1);
pers_len_set_region = zeros(size(tp_train_set,1),1);

ts_warmup = zeros(warmup_r_step_length,dim);
ts_test_target = zeros(predict_r_step_length,dim);
for tp_i = 1:size(tp_train_set,1)
    tp = tp_train_set(tp_i,:);

    start_step_i = randi(size(data_all,2));
    train_data_sample = zeros(size(data_all,2),dim);
    train_data_sample(:,:) = data_all( train_select_set(tp_i),:,:);
    train_data_sample = repmat(train_data_sample,...
        [ round((warmup_r_step_length+predict_r_step_cut+predict_r_step_length)/size(data_all,2))+5,1]);
    ts_warmup(:,:) = train_data_sample( ...
        start_step_i:start_step_i-1+warmup_r_step_length,:);
    ts_test_target(:,:) = train_data_sample( ...
        start_step_i-1+warmup_r_step_length +1 +predict_r_step_cut :...
        start_step_i-1+warmup_r_step_length +1 +predict_r_step_cut + predict_r_step_length-1,:);

    flag_r_test = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
    predict_test = func_STP_predict_uniformedST(ts_warmup,tp,W_in,W_r,W_out,flag_r_test);

    % calculate different types of errors
    % rmse
    error = predict_test - ts_test_target;
    error = error(1:rmse_len,:);
    se_ts = sum( error.^2 ,2);
    rmse_set(tp_i) = sqrt(mean(se_ts));
    
    % persiting time til +- 5%
    temp_len = 0;
    dims_ud = zeros(dim,3);
    for d_i = 1:dim
        dims_ud(d_i,1) = max(ts_test_target(:,d_i));
        dims_ud(d_i,2) = min(ts_test_target(:,d_i));
        dims_ud(d_i,3) = dims_ud(d_i,1) - dims_ud(d_i,2);
    end
    % thres 1
    for t_i = 1:predict_r_step_length
        bool_pers_temp = 1;
        for d_i = 1:dim
            error_temp = abs(predict_test(t_i,d_i)-ts_test_target(t_i,d_i));
            %if (predict_test(t_i,d_i)-dims_ud(d_i,1)) > pers_thres*dims_ud(d_i,3) ...
            %        || (dims_ud(d_i,2)-predict_test(t_i,d_i)) > pers_thres*dims_ud(d_i,3)
            if error_temp > pers_thres1*dims_ud(d_i,3)
                bool_pers_temp = 0;
                break
            end
        end
        if bool_pers_temp == 0
            break
        end
        temp_len = temp_len + 1;
    end
    pers_len_set_new1(tp_i) = temp_len;
    % thres 2
    for t_i = 1:predict_r_step_length
        bool_pers_temp = 1;
        for d_i = 1:dim
            error_temp = abs(predict_test(t_i,d_i)-ts_test_target(t_i,d_i));
            %if (predict_test(t_i,d_i)-dims_ud(d_i,1)) > pers_thres*dims_ud(d_i,3) ...
            %        || (dims_ud(d_i,2)-predict_test(t_i,d_i)) > pers_thres*dims_ud(d_i,3)
            if error_temp > pers_thres2*dims_ud(d_i,3)
                bool_pers_temp = 0;
                break
            end
        end
        if bool_pers_temp == 0
            break
        end
        temp_len = temp_len + 1;
    end
    pers_len_set_new2(tp_i) = temp_len;


    % persiting time til +- 5%, region
    temp_len = 0;
    dims_ud = zeros(dim,3);
    for d_i = 1:dim
        dims_ud(d_i,1) = max(ts_test_target(:,d_i));
        dims_ud(d_i,2) = min(ts_test_target(:,d_i));
        dims_ud(d_i,3) = dims_ud(d_i,1) - dims_ud(d_i,2);
    end
    for t_i = 1:predict_r_step_length
        bool_pers_temp = 1;
        for d_i = 1:dim
            %error_temp = abs(predict_test(t_i,d_i)-ts_test_target(t_i,d_i));
            if (predict_test(t_i,d_i)-dims_ud(d_i,1)) > pers_thres1*dims_ud(d_i,3) ...
                    || (dims_ud(d_i,2)-predict_test(t_i,d_i)) > pers_thres1*dims_ud(d_i,3)
            %if error_temp > pers_thres*dims_ud(d_i,3)
                bool_pers_temp = 0;
                break
            end
        end
        if bool_pers_temp == 0
            break
        end
        temp_len = temp_len + 1;
    end
    pers_len_set_region(tp_i) = temp_len;

    %%
    %
    if plot_rand_num ~= 0
        if randi(plot_rand_num) == 1
            figure()
            set(gcf,'color','white')
            subplot(2,1,1)
            hold on
            plot(predict_test(:,1),'b')
            plot(ts_test_target(:,1),'r--')
            hold off
            subplot(2,1,2)
            hold on
            plot(predict_test(:,2),'b')
            plot(ts_test_target(:,2),'r--')
            hold off
        end
    end
    %


end


filename_save = ['loop_switch_2_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

rand_start_len = 2e4;
warmup_r_step_length = 2.5e3;

predict_r_step_length_1 = 4.5e3;
predict_r_step_length_2 = 4.5e3;

predict_r_step_length_cut = 2e3;

repeat_num = 100;
plot_dim = 1;
plot_ratio = 1;


result_all = zeros(6,6,repeat_num,2);

%% loop
tic
for tp_i_1 = 1:6
    for tp_i_2 = 1:6
        if tp_i_1 == tp_i_2
            continue
        end
        
        at_i_1 = tp_order(tp_i_1);
        at_i_2 = tp_order(tp_i_2);
        
        repeat_temp = zeros(repeat_num,2);
        parfor repeat_i = 1:repeat_num
            rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
            
            tp_1 = tp_train_set(tp_i_1,:);
            tp_2 = tp_train_set(tp_i_2,:);
            
            
            %% warm up
            start_step_i = randi(rand_start_len);
            ts_warmup = zeros(warmup_r_step_length,dim);
            ts_warmup(:,:) = data_all(tp_i_1,...
                4e3+start_step_i:4e3+start_step_i-1+warmup_r_step_length,:);
            
            %% predict
            flag_r = [n dim a warmup_r_step_length ...
                predict_r_step_length_1 predict_r_step_length_2];
            predict_r = func_STP_predict_switch(ts_warmup,tp_1,tp_2,W_in,res_net,P,flag_r);
            
            bool_before = func_check_S6_3( predict_r(...
                predict_r_step_length_cut+1:predict_r_step_length_1,:) , at_i_1 );
            
            bool_after = func_check_S6_3( predict_r(...
                predict_r_step_length_1 + ...
                predict_r_step_length_cut+1:end,:) , at_i_2 );
            
            repeat_temp(repeat_i,:) = [bool_before, bool_after];
        end
        result_all(tp_i_1,tp_i_2,:,:) = repeat_temp;
        
        fprintf(['From No.' num2str(tp_i_1) ' to No.' num2str(tp_i_2) ...
            ' is done\n'])
        toc
    end
end


save(filename_save)




%% parameters
warmup_r_step_length = 250;
warmup_attractor_i = 3;

rand_start_len = 2e4;
predict_r_step_cut = 50;  % gen 20
predict_r_step_length = 400; % gen 20

%% main
tic
start_step_i = randi(rand_start_len);
ts_warmup = zeros(warmup_r_step_length,dim);
ts_warmup(:,:) = data_all(warmup_attractor_i,...
    4e3+start_step_i:4e3+start_step_i-1+warmup_r_step_length,:);

flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
[predict_r,r_hidden] = func_STP_predict_uniformedST_hidden(ts_warmup,0,W_in,res_net,P,flag_r);
toc

%% plot
figure()
plot3(predict_r(:,1),predict_r(:,2),predict_r(:,3))
set(gcf,'color','white')
%clear data_all

%{
figure()
imagesc(r_hidden')
title(['attractor No.' num2str(warmup_attractor_i)])
set(gcf,'color','white')

figure()
histogram(r_hidden)
ylim([0 1.6e5])
title(['attractor No.' num2str(warmup_attractor_i)])
set(gcf,'color','white')
%}



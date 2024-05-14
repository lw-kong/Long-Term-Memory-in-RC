addpath('\klw\Research\Functions');

rand_start_len = 1e4;
warmup_r_step_length = 2e3;

predict_r_step_cut = 1e3;
predict_r_step_length = 2e4;
predict_r_step_length = 5e3;
%predict_r_step_length = 2e5;

%tp_i = 2;
tp = tp_train_set(tp_i,:);

attractor_i = tp_order(tp_i);

plot_real = 1;

rng('shuffle');
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict
start_step_i = randi(rand_start_len);
ts_warmup = zeros(warmup_r_step_length,dim);
ts_warmup(:,:) = data_all(tp_i,...
    4e3+start_step_i:4e3+start_step_i-1+warmup_r_step_length,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
predict_r = func_STP_predict(ts_warmup,tp,W_in,res_net,P,flag_r);

toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot
label_font_size = 12;
ticks_font_size = 12;

%{
plot_axis = [ -3,3,-3,3,-3,3; % Lorenz
    -2,2,-2,2,-1,9; % Rossler
    -4,4,-4,4,-4,5;
    -1.5,5.5,-2,2.5,-2.5,2; % HR
    -2,3.5,-2,3.5,-2,3.5; % Food chain
    -4,4,-1,5,-2,2];
%}

% S6_3
plot_axis = [ -3,3,-3,3,-3,3;
    -3,3,-3,3,-3,3;
    -3,3,-3,3,-3,3; % Lorenz
    -2,2,-2,2,-1,9; % Rossler
    -2,3.5,-2,3.5,-2,3.5; % Food chain
    -1.5,5.5,-2,2.5,-2.5,2]; % HR


c_med_blue = [114,158,206]/255;
c_med_red = [237,102,93]/255;

figure()
set(gcf,'color','white')

if plot_real == 1
    plot_real = zeros(predict_r_step_length,3);
    plot_real(:,:) = data_all(attractor_i,...
        4e3+start_step_i:4e3+start_step_i-1+predict_r_step_length,:);
    
    subplot(2,1,1)
    plot3(plot_real(:,1), plot_real(:,2), plot_real(:,3),'Color',c_med_blue)
    %view([1 -1 0.5])
    view([1 -1 1])
    axis(plot_axis(attractor_i,:))
    title( ['real, system No.' num2str(attractor_i)])
end

subplot(2,1,2)
plot3(predict_r(:,1), predict_r(:,2), predict_r(:,3),'Color',c_med_red)
%view([1 -1 0.5])
view([1 -1 1])
axis(plot_axis(attractor_i,:))
title( ['predicted, system No.' num2str(attractor_i)])







%{
figure(),plot(plot_real(:,1))
figure(),plot(plot_real(:,2))
figure(),plot(plot_real(:,3))
%}

%{
figure(),plot(predict_r(:,1)),set(gcf,'color','white')
figure(),plot(predict_r(:,2)),set(gcf,'color','white')
figure(),plot(predict_r(:,3)),set(gcf,'color','white')
%}
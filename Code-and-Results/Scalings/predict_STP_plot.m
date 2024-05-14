addpath('\klw\Research\Functions');

rand_start_len = 1e4;
warmup_r_step_length = 2e3;

predict_r_step_cut = 1e3;
%predict_r_step_length = 2e4;
predict_r_step_length = 5e3;
%predict_r_step_length = 2e5;


plot_real = 1;

rng('shuffle');
tic;
plot_predict_all = zeros(6,predict_r_step_length,dim);
for tp_i = 1:6
    tp = tp_train_set(tp_i,:);
    attractor_i = tp_order(tp_i);

    % predict
    start_step_i = randi(rand_start_len);
    ts_warmup = zeros(warmup_r_step_length,dim);
    ts_warmup(:,:) = data_all(tp_i,...
        4e3+start_step_i:4e3+start_step_i-1+warmup_r_step_length,:);

    flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
    predict_r = func_STP_predict(ts_warmup,tp,W_in,res_net,P,flag_r);
    plot_predict_all(tp_i,:,:) = predict_r;
    toc
end
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

%{
plot_real_all = zeros(6,predict_r_step_length,dim);
for tp_i = 1:6
    tp = tp_train_set(tp_i,:);
    attractor_i = tp_order(tp_i);

    plot_real_all(tp_i,:,:) = data_all(attractor_i,...
        4e3+start_step_i:4e3+start_step_i-1+predict_r_step_length,:);
end
%}

c_med_blue = [114,158,206]/255;
c_med_red = [237,102,93]/255;
plot_line_width_1 = 1;
plot_line_width_2 = 0.6;
figure()
set(gcf,'color','white')
for tp_i = 1:6
    tp = tp_train_set(tp_i,:);
    attractor_i = tp_order(tp_i);

    plot_real = zeros(predict_r_step_length,3);
    plot_real(:,:) = data_all(attractor_i,...
        4e3+start_step_i:4e3+start_step_i-1+predict_r_step_length,:);

    if tp_i <= 3
        plot_i = tp_i;
    else
        plot_i = tp_i + 3;
    end
    if tp_i <= 2
        plot_line_width = plot_line_width_1;
    else
        plot_line_width = plot_line_width_2;
    end

    subplot(4,3,plot_i)
    plot3(plot_real(:,1), plot_real(:,2), plot_real(:,3),...
        'Color',c_med_blue,'LineWidth',plot_line_width)
    %view([1 -1 0.5])
    view([1 -1 1])
    axis(plot_axis(attractor_i,:))
    %title( ['real, system No.' num2str(attractor_i)])


    subplot(4,3,plot_i+3)
    plot_predict_temp = zeros(predict_r_step_length,3);
    plot_predict_temp(:,:) = plot_predict_all(tp_i,:,:);
    plot3(plot_predict_temp(:,1), plot_predict_temp(:,2), plot_predict_temp(:,3),...
        'Color',c_med_red,'LineWidth',plot_line_width)
    %view([1 -1 0.5])
    view([1 -1 1])
    axis(plot_axis(attractor_i,:))
    %title( ['predicted, system No.' num2str(attractor_i)])
end


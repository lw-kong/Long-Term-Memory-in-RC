addpath('\klw\Research\Functions');

rand_start_len = 2e4;
warmup_r_step_length = 2.5e3;

predict_r_step_length_1 = 1e4;
predict_r_step_length_2 = 1e4;

predict_r_step_1_cut = 0;

tp_i_1 = 1;
tp_i_2 = 6;


tp_1 = tp_train_set(tp_i_1,:);
tp_2 = tp_train_set(tp_i_2,:);

rng('shuffle');
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict
start_step_i = randi(rand_start_len);
ts_warmup = zeros(warmup_r_step_length,dim);
ts_warmup(:,:) = data_all(tp_i_1,...
    4e3+start_step_i:4e3+start_step_i-1+warmup_r_step_length,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag_r = [n dim a warmup_r_step_length ...
    predict_r_step_length_1 predict_r_step_length_2];
[predict_r,predict_hidden] = func_STP_predict_switch_hidden(...
    ts_warmup,tp_1,tp_2,W_in,res_net,P,flag_r);

toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot
label_font_size = 12;
ticks_font_size = 12;

plot_axis = [ -3,3,-3,3,-3,3;
    -2,2,-2,2,-1,9;
    -4,4,-4,4,-4,4;
    -1.5,5.5,-2,2.5,-2.5,2;
    -2,3.5,-2,3.5,-2,3.5;
    -2.5,2,-2.5,2,-2.5,2];

%
plot_dim = 1;

figure()
set(gcf,'color','white')
plot(predict_r(:,plot_dim))
title(['From System No.' num2str(tp_i_1) ' to System No.' num2str(tp_i_2)])
%
    
%{
figure()
set(gcf,'color','white')
for plot_dim = 1:dim
    subplot(dim,1,plot_dim)
    plot(predict_r(:,plot_dim))
    title(['From System No.' num2str(tp_i_1) ' to System No.' num2str(tp_i_2)])
end
%}

%{
figure()
set(gcf,'color','white')
plot_dim_max = 5;
for plot_dim = 1:plot_dim_max
    subplot(plot_dim_max,1,plot_dim)
    plot_dim_rand = randi(n);
    plot(predict_hidden(:,plot_dim_rand))
    title({['From System No.' num2str(tp_i_1) ' to System No.' num2str(tp_i_2)],...
        [ 'hidden node ' num2str(plot_dim_rand) ]})
end
%}

figure()
set(gcf,'color','white')
subplot(1,2,1)
imagesc(predict_hidden(predict_r_step_1_cut+1:end,1:80)')
xlabel('steps')
ylabel('node')

subplot(1,2,2)
imagesc(tanh(10*W_in(1:80,end)))
ylabel('node')
title('tanh(10*Win)')
%


figure()
set(gcf,'color','white')
scatter( W_in(:,end), median(predict_hidden(...
    0.2*predict_r_step_length_1:predict_r_step_length_1,:)) )
hold on
plot( sort(W_in(:,end)), tanh(  tp_1*sort(W_in(:,end)) ) , 'Color','r','LineWidth',3)
xlabel('W_{b}')
ylabel('hidden state median value')
title(['System No.' num2str(tp_i_1)])
box on
hold off

figure()
set(gcf,'color','white')
scatter( W_in(:,end), median(predict_hidden(...
    predict_r_step_length_1 + 0.2*predict_r_step_length_2:end,:)) )
hold on
plot( sort(W_in(:,end)), tanh(  tp_2*sort(W_in(:,end)) ) , 'Color','r','LineWidth',3)
xlabel('W_{b}')
ylabel('hidden state median value')
title(['System No.' num2str(tp_i_2)])
ylim([-1 1])
box on
hold off

%{
figure()
set(gcf,'color','white')
if tp_i ~=6
    if plot_real == 1
        plot_real = zeros(predict_r_step_length,3);
        plot_real(:,:) = data_all(tp_i,...
            4e3+start_step_i:4e3+start_step_i-1+predict_r_step_length,:);
        
        subplot(2,1,1)
        plot3(plot_real(:,1), plot_real(:,2), plot_real(:,3))
        view([1 -1 0.5])
        axis(plot_axis(tp_i,:))
        title( ['real, system No.' num2str(tp_i)])
    end
    
    subplot(2,1,2)
    plot3(predict_r(:,1), predict_r(:,2), predict_r(:,3))
    view([1 -1 0.5])
    axis(plot_axis(tp_i,:))
    title( ['predicted, system No.' num2str(tp_i)])
    
elseif tp_i == 6
    MG_tau = 17;
    
    if plot_real == 1
        plot_real = zeros(predict_r_step_length,1);
        plot_real(:,:) = data_all(tp_i,...
            4e3+start_step_i:4e3+start_step_i-1+predict_r_step_length,1);
        
        subplot(2,1,1)
        plot3(plot_real(1:end-2*MG_tau,1),...
            plot_real(1+MG_tau:end-MG_tau,1), ...
            plot_real(1+2*MG_tau:end,1))
        view([1 0.5 1])
        axis(plot_axis(tp_i,:))
        title( ['real, system No.' num2str(tp_i)])
    end
    
    subplot(2,1,2)
    plot3(predict_r(1:end-2*MG_tau,1),...
        predict_r(1+MG_tau:end-MG_tau,1),...
        predict_r(1+2*MG_tau:end,1))
    view([1 0.5 1])
    axis(plot_axis(tp_i,:))
    title( ['predicted, system No.' num2str(tp_i)])
end
%}



%{
figure(),plot(plot_real(:,1))
figure(),plot(plot_real(:,2))
figure(),plot(plot_real(:,3))
%}
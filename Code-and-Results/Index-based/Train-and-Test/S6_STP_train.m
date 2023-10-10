
addpath 'D:\Research\Reservoir\MultiMemory_InitHideenState\Wave-3-2022.5\GitHub\Google Drive\Data'

dim = 3;

% load('opt_S6_1smaller_20_20220712T112045_642.mat')
n = 1e3;
eig_rho = 0.78;
W_in_a = 0.85;
a = 0.37;
beta = 1 * 10^(-7.5); 
k = 210;
noise_a = 10^(-3.1);
tp_W = 1.12;
tp_bias = -1.08;


tp_order = 1:6;
tp_train_set = tp_W*((1:6)'-3.5)+tp_bias;
dim_set = [3,3,3,3,3,3];


rand_start_len = 1e4;
train_r_step_length = 6e3;
validate_r_step_length = 200;


W_in_type = 1;

bo = 5;


rng('shuffle');
tic;

%% preparing training data
train_data_length = train_r_step_length + validate_r_step_length + 10;
train_data = zeros(length(tp_train_set), train_data_length,dim+1); % data that goes into reservior_training

load('data_S6_3.mat')
for tp_i = 1:length(tp_train_set)
    tp = tp_train_set(tp_i);
    
    start_step_i = randi(rand_start_len);
    train_data(tp_i,:,1:dim) = data_all( tp_order(tp_i),...
        4e3+start_step_i:4e3+start_step_i-1+train_data_length,:);
    
        
    train_data(tp_i,:,dim+1) = tp * ones(train_data_length,1);    %% system sensitive
end


%% train
rmse_min = 1e5;
for bo_i = 1:bo
    fprintf('preparing training data...\n');
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % train
    fprintf('training...\n');
    flag_r_train = [n k eig_rho W_in_a a beta 0 train_r_step_length validate_r_step_length...
        1 dim noise_a];
    [rmse,W_in_temp,res_net_temp,P_temp,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_STP_train_NullDim_noise(...
        train_data,tp_train_set,dim_set,flag_r_train,W_in_type,2,1);
    fprintf('attempt rmse = %f\n',rmse)
    
    if rmse < rmse_min
        W_in = W_in_temp;
        res_net = res_net_temp;
        P = P_temp;
        t_validate = t_validate_temp;
        x_real = x_real_temp;
        x_validate = x_validate_temp;
        rmse_min = rmse;
    end
    
    fprintf('%f is done\n',bo_i/bo)
    toc;
end

fprintf('best rmse = %f\n',rmse_min)

plot_dim = 1; % change the ylabel
for tp_i = 1:length(tp_train_set)
    figure('Name','Reservoir Predict')
    set(gcf,'color','white')
    
    subplot(2,1,1)
    hold on
    plot(t_validate,x_real(tp_i,:,plot_dim));
    plot(t_validate,x_validate(tp_i,:,plot_dim),'--');
    xlabel('time');
    ylabel('V');
    title(['system No.' num2str( tp_i )])
    box on
    hold off
    
    subplot(2,1,2)
    hold on
    plot(t_validate,abs(x_validate(tp_i,:,plot_dim)-x_real(tp_i,:,plot_dim))/...
        ( max(x_real(tp_i,:,plot_dim)) - min(x_real(tp_i,:,plot_dim)) ) )
    line([t_validate(1) t_validate(end)],[0.05 0.05],'Color','black','LineStyle','--')
    xlabel('time');
    ylabel('relative error')
    box on
    hold off
end


%
% plotting training data
figure('Name','Training Data','Position',[50 50 480 390]);
hold on
for tp_i = 1:length(tp_train_set)
    
    
    plot(train_data(tp_i,:,1),train_data(tp_i,:,2));
    xlabel('x');
    ylabel('y');
    %title(['training data at' newline 'tp = ' num2str( para_train_set(tp_i),8 )]);
    set(gcf,'color','white')
    
end
box on
hold off

%
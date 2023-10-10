


dim = 3;
% load('opt_S6_noTP_30_20220609T212726_60.mat')
%n = 2e3;
eig_rho = 1.47; % the larger, the longer the memory
W_in_a = 1.13;
a = 1;     % the smaller, the longer the memory
beta = 1 * 10^(-6.4);
noise_a = 10^(-2.9);


n = 2400;
k = round(10/2e3*n);

%tp_order = [1,2,4];
tp_order = 1:6;
tp_train_set = zeros(length(tp_order),1);
dim_set = [3,3,3,3,3,3];



rand_start_len = 1e4;
train_r_step_length = 6e3;
validate_r_step_length = 300;


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
        func_STP_train_NullDim_50(...
        train_data,tp_train_set,dim_set,flag_r_train,W_in_type,2,1,2);
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

clear res_net_temp W_in_temp P_temp t_validate_temp x_real_temp x_validate_temp rmse

%% plot
fprintf('best rmse = %f\n',rmse_min)

plot_dim = 3; % change the ylabel
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
    line([t_validate(1) t_validate(end)],[0.05 0.05])
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

%
% plotting training data
figure('Name','Training Data','Position',[50 50 480 390]);
for tp_i = 1:length(tp_train_set)  
    plot3(train_data(tp_i,:,1),train_data(tp_i,:,2),train_data(tp_i,:,3))
    hold on
    xlabel('x')
    ylabel('y')
    zlabel('z')
    %title(['training data at' newline 'tp = ' num2str( para_train_set(tp_i),8 )]);
    set(gcf,'color','white')    
end
box on
hold off
%
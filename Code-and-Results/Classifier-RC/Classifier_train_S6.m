% chose one of the best bo training results
% different tp_set and ode_parameter_set
% warm up
% no return map validation
% control parameter is 1 dim

addpath('\klw\Research\Functions');
addpath('\klw\Research\Reservoir\PredictBehaviorUnderDiffPara\STP_reorganize_phase');


dim = 3;

% load('opt_S6_noTP_14_20220405T180052_793.mat')
%n = 1.5e3;
n = 2e3;
eig_rho = 1.18;
W_in_a = 1.27;
a = 0.44;
beta = 1 * 10^(-7.6);
k = round(10/1.5e3*n);
noise_a = 10^(-3);

tp_num = 6;
train_repeat_num = 10;
tp_train_set = repmat(1:tp_num,[1,train_repeat_num]);


rand_start_len = 1.5e4;
train_r_step_length = 500;
validate_r_step_cut = 100;
validate_r_step_length = 500;


W_in_type = 1;
bo = 1;


rng('shuffle');
tic;

%% preparing training data
train_data_length = train_r_step_length + validate_r_step_cut + validate_r_step_length + 10;
train_input = zeros(length(tp_train_set), train_data_length,dim); % data that goes into reservior_training
train_target = zeros(length(tp_train_set), train_data_length,tp_num);

load('data_S6_2.mat')
for trial_i = 1:length(tp_train_set)
    tp = tp_train_set(trial_i);
    
    start_step_i = randi(rand_start_len);
    train_input(trial_i,:,1:dim) = data_all(tp,...
        4e3+start_step_i:4e3+start_step_i-1+train_data_length,:);
    
    train_target(trial_i,:,tp) = ones(train_data_length,1);
end


%% train
rmse_min = 1e5;
for bo_i = 1:bo
    fprintf('preparing training data...\n');
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % train
    fprintf('training...\n');
    flag_r_train = [n k eig_rho W_in_a a beta 0 train_r_step_length validate_r_step_length...
        validate_r_step_cut 1 dim noise_a];
    [rmse,W_in_temp,res_net_temp,P_temp,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_train_classifier(...
        train_input,train_target,flag_r_train,W_in_type,2,1);
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


figure('Name','Reservoir Predict')
set(gcf,'color','white')
for trial_i = 1:length(tp_train_set)   
    subplot(train_repeat_num,6,trial_i)
    plot_temp = reshape(x_validate(trial_i,:,:),size(x_validate,2),size(x_validate,3));

    score_temp = mean(plot_temp);
    score_temp = exp(10*score_temp);
    score_temp = score_temp/sum(score_temp);

    imagesc(plot_temp')
    clim([0,1])
    xlabel('steps');
    ylabel('index of attractors');
    %title(['Validate with No.' num2str( tp_train_set(trial_i) )])
    title(['score: ' num2str(score_temp(tp_train_set(trial_i)))])
    box on
end






%{
% plotting training data
figure('Name','Training Data','Position',[50 50 480 390]);
hold on
for trial_i = 1:length(tp_train_set)  
    plot(train_input(trial_i,:,1),train_input(trial_i,:,2));
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
for trial_i = 1:length(tp_train_set)  
    plot3(train_input(trial_i,:,1),train_input(trial_i,:,2),train_input(trial_i,:,3))
    hold on
    xlabel('x')
    ylabel('y')
    zlabel('z')
    %title(['training data at' newline 'tp = ' num2str( para_train_set(tp_i),8 )]);
    set(gcf,'color','white')    
end
box on
hold off
%}


%{
figure()
plot(train_input(4,1:500,1))
xlabel('steps')
set(gcf,'color','white') 
%}
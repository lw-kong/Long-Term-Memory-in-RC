
addpath '..'
load('data_st_color_1.mat')


%% rmse
rmse_val_threshold = 0.5;
rmse_tolerance_rate = 0.02;
rmse_tolerance_rate = 0;

rmse_c_indx = floor(rmse_tolerance_rate * length(tp_order)) + 1;

plot_rmse_reorg = zeros(length(n_set),repeat_num);
rmse_set_temp = zeros(length(tp_order),1);
for n_i = 1:length(n_set)
    for repeat_i = 1:repeat_num
        rmse_set_temp(:) = result_rmse_set(n_i,repeat_i,:);
        rmse_set_temp = sort(rmse_set_temp,'descend');
        plot_rmse_reorg(n_i,repeat_i) = rmse_set_temp(rmse_c_indx);
    end
end
plot_temp = -sign(plot_rmse_reorg-rmse_val_threshold);
plot_temp = max(plot_temp,0);
plot_success_curve_rmse2 = mean(plot_temp,2);


figure('Position',[300,300,450*2,300*2])
set(gcf,'color','white')

subplot(2,2,1)
%scatter(n_set,plot_success_curve)
scatter(n_set,plot_success_curve_rmse2,40,'filled',...
    'MarkerEdgeColor','none','MarkerFaceColor',c_med_blue)
line(n_set,plot_success_curve_rmse2)
line([min(n_set),max(n_set)],[0.5,0.5],'Color','black')
box on
xlabel('N')
ylabel('success rate')
ylim([0,1])
title('RMSE')
grid on


%% pers_len region
len_threshold_region = 800;
plot_success_curve_perslen = zeros(length(n_set),1);
for n_i = 1:length(n_set)
    plot_temp_1 = min(result_pers_set_region(n_i,:,:),[],3);
    plot_temp = sign(plot_temp_1-len_threshold_region);
    plot_temp = max(plot_temp,0);
    plot_success_curve_perslen(n_i) = mean(plot_temp);
end


%scatter(n_set,plot_success_curve)
subplot(2,2,2)
scatter(n_set,plot_success_curve_perslen,40,'filled',...
    'MarkerEdgeColor','none','MarkerFaceColor',c_med_blue)
line(n_set,plot_success_curve_perslen)
line([min(n_set),max(n_set)],[0.5,0.5],'Color','black')
box on
xlabel('N')
ylabel('success rate')
ylim([0,1])
title('pers len region')
grid on

%% pers_len new
len_threshold_new = 400;
plot_success_curve_perslen_new1 = zeros(length(n_set),1);
for n_i = 1:length(n_set)
    plot_temp_1 = min(result_pers_set_new1(n_i,:,:),[],3);
    plot_temp = sign(plot_temp_1-len_threshold_new);
    plot_temp = max(plot_temp,0);
    plot_success_curve_perslen_new1(n_i) = mean(plot_temp);
end


subplot(2,2,3)
%scatter(n_set,plot_success_curve)
scatter(n_set,plot_success_curve_perslen_new1,40,'filled',...
    'MarkerEdgeColor','none','MarkerFaceColor',c_med_blue)
line(n_set,plot_success_curve_perslen_new1)
line([min(n_set),max(n_set)],[0.5,0.5],'Color','black')
box on
xlabel('N')
ylabel('success rate')
ylim([0,1])
title('pers len new1')
grid on

plot_success_curve_perslen_new2 = zeros(length(n_set),1);
for n_i = 1:length(n_set)
    plot_temp_1 = min(result_pers_set_new2(n_i,:,:),[],3);
    plot_temp = sign(plot_temp_1-len_threshold_new);
    plot_temp = max(plot_temp,0);
    plot_success_curve_perslen_new2(n_i) = mean(plot_temp);
end


subplot(2,2,4)
%scatter(n_set,plot_success_curve)
scatter(n_set,plot_success_curve_perslen_new2,40,'filled',...
    'MarkerEdgeColor','none','MarkerFaceColor',c_med_blue)
line(n_set,plot_success_curve_perslen_new2)
line([min(n_set),max(n_set)],[0.5,0.5],'Color','black')
box on
xlabel('N')
ylabel('success rate')
ylim([0,1])
title('pers len new2')
grid on

%% reconError

figure('Position',[300,300,450,300])
set(gcf,'color','white')
hold on
errorbar(n_set,mean(result_reconError,2),std(result_reconError,0,2))
scatter(n_set,mean(result_reconError,2))
xlabel('N')
ylabel('Reconstruction Error')
title('recon error')
box on
hold off

%% W_out
plot_W_out = zeros(length(n_set),repeat_num);
for n_i = 1:length(n_set)
    for repeat_i = 1:repeat_num
        W_out_temp = result_W_out{n_i,repeat_i};
        plot_W_out(n_i,repeat_i) = norm(W_out_temp);
    end
end

figure('Position',[300,300,450,300])
set(gcf,'color','white')
hold on
errorbar(n_set,mean(plot_W_out,2),std(plot_W_out,0,2))
scatter(n_set,mean(plot_W_out,2))
xlabel('N')
ylabel('norm( W out )')
title('norm( W out )')
box on
hold off

%{
save('save_plot_7oh_2_128.mat',...
    "n_set","tp_order","tp_W","tp_bias","noise_a","repeat_num","paras_error","filename_save",...
    "rmse_val_threshold","rmse_tolerance_rate","len_threshold",...
    "plot_success_curve_rmse2","plot_success_curve_perslen","result_reconError","plot_W_out")
%}

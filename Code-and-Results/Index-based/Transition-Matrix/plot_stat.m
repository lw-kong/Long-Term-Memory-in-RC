


plot_result = ones(length(tp_order)); % success rate
denom_result = zeros(length(tp_order));
for tp_i_1 = 1:length(tp_order)
    for tp_i_2 = 1:length(tp_order)
        if tp_i_1 == tp_i_2
            continue
        end
        
        denom_temp = 0;
        nom_temp = 0;
        for repeat_i = 1:repeat_num
            if result_all(tp_i_1,tp_i_2,repeat_i,1) == 1
                denom_temp = denom_temp +1;
                if result_all(tp_i_1,tp_i_2,repeat_i,2) == 1
                    nom_temp = nom_temp + 1;
                end
            end
        end
        
        plot_result(tp_i_1,tp_i_2) = nom_temp / denom_temp;
        denom_result(tp_i_1,tp_i_2) = denom_temp;
    end
end
            
figure()
set(gcf,'color','white')
imagesc(plot_result,[0.2,1])
xlabel('destination')
ylabel('starting point')
colormap('bone')
colorbar
title(filename_load)

%save('save_plot_S6_659_2.mat','denom_result','plot_result','repeat_num')



%
figure()
set(gcf,'color','white')
imagesc(denom_result/repeat_num,[0,1])
xlabel('destination')
ylabel('starting point')
title(['denom ' filename_load])
colormap('bone')
colorbar
%
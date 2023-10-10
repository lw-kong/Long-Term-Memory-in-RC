
step_length_total = 200;
num_samples = 1e4;
dim = 3;

warning('off')

num_points_range = [2,3,4,5];
step_block = 200;
num_block = round(step_length_total/step_block)+1;
data_range = [-1,1]; % renormalize later

plot_bool = 1;
if num_samples > 25
    plot_bool = 0;
end

data_s0 = zeros(num_samples,step_length_total,dim);
tic
for sample_i = 1:num_samples
    data_sample = zeros(step_block,dim);
    for d_i = 1:dim
    num_points = num_points_range(randi(length(num_points_range)));
    points_set_temp = zeros(num_points,2);
    for point_i = 1:num_points
        points_set_temp(point_i,2) = ...
            abs(data_range(2)-data_range(1))*rand+min(data_range);
        t_range_min_temp = (point_i-1)*step_block/num_points ...
            + step_block/num_points/4;
        t_range_width_temp = step_block/num_points/2;
        points_set_temp(point_i,1) = ...
            round( t_range_width_temp * rand + t_range_min_temp );
    end

    fit = @(b,x)  b(1).*sin(2*pi*x/step_block +b(2)) ...
        + b(3).*sin(2*2*pi*x/step_block +b(4)) ...
        + b(5).*sin(3*2*pi*x/step_block +b(6)) ...
        + b(7).*sin(4*2*pi*x/step_block +b(8));
        %...
        %+ b(9).*sin(5*2*pi*x/step_block +b(10));    % Function to fit
    fcn = @(b) sum((fit(b,points_set_temp(:,1)) - points_set_temp(:,2)).^2);                              % Least-Squares cost function
    opts = optimset('Display','off');
    s_temp = fminsearch(fcn, zeros(8,1),opts);                       % Minimise Least-Squares
    data_temp = fit(s_temp,1:step_block);
    data_sample(:,d_i) = (data_temp-mean(data_temp))/std(data_temp);
    end
    %
    figure()
    set(gcf,'color','white')
    scatter(points_set_temp(:,1),points_set_temp(:,2))
    hold on
    plot(1:step_block,data_temp)
    box on
    hold off
    %

    
    %% plot
    %
    if plot_bool == 1
    figure()
    set(gcf,'color','white')
    for d_i = 1:dim
        subplot(2,2,d_i)
        plot(data_sample(:,d_i))
    end
    subplot(2,2,4)
    plot3(data_sample(:,1),data_sample(:,2),data_sample(:,3))
    end
    %


    data_sample = repmat(data_sample,[num_block,1]);
    data_s0(sample_i,:,:) = data_sample(1:step_length_total,:);
    if mod(sample_i,100) == 0
        fprintf('%f is done\n',sample_i/num_samples)
        toc
    end
end

num_plot = 25;
for plot_i = 1:min([num_samples,num_plot])
    data_sample = zeros(step_block,dim);
    data_sample(:,:) = data_s0(randi(num_samples),1:step_block,:);

    figure()
    set(gcf,'color','white')
    for d_i = 1:dim
        subplot(2,2,d_i)
        plot(data_sample(:,d_i))
    end
    subplot(2,2,4)
    plot3(data_sample(:,1),data_sample(:,2),data_sample(:,3))

end



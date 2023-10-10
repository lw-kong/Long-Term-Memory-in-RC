figure()
set(gcf,'color','white')
for plot_i = 1:6
   subplot(2,3,plot_i)
   plot3(data_all(plot_i,2e4+1:2e4+5e3,1),...
       data_all(plot_i,2e4+1:2e4+5e3,2),...
       data_all(plot_i,2e4+1:2e4+5e3,3))  
   view([1 -1 1])
   %xlim([-2.5,2.5])
   %ylim([-2.5,2.5])
   %zlim([-2.5,2.5])
end
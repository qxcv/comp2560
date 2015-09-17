% display the detected pose sequence
function show_pose_sequence(data_path, frames, detected_pose_seq, parent)
figno=1023; figure(figno);clf(figno); 

for n = 1:length(frames)   
    %video_showskeleton2(img{n}, detected_pose_seq(n,:));axis off;
    video_showskeleton2([data_path frames(n).name], detected_pose_seq(n,:), parent);
    axis off;
    pause(1);
end
end

function video_showskeleton2(frame, boxes, parent)
 img = imread(frame);
 imagesc(img); hold on; 
  
  for i = 1:length(parent)
    x1(:,i) = boxes(:,1+(i-1)*4);
    y1(:,i) = boxes(:,2+(i-1)*4);
    x2(:,i) = boxes(:,3+(i-1)*4);
    y2(:,i) = boxes(:,4+(i-1)*4);
  end
  x = (x1 + x2)/2;
  y = (y1 + y2)/2;  
  
  hold on;
  % line([x(1,2), x(1,4)], [y(1,2), y(1,4)],'color', 'r','linewidth', 3);
  line([x(1,3), x(1,5)], [y(1,3), y(1,5)],'color', 'r','linewidth', 3);
  % line([x(1,4), x(1,6)], [y(1,4), y(1,6)],'color', 'b','linewidth', 3);
  line([x(1,5), x(1,7)], [y(1,5), y(1,7)],'color', 'b','linewidth', 3);
  % line([x(1,7), x(1,9)], [y(1,7), y(1,9)],'color', 'g','linewidth', 3);
  line([x(1,11), x(1,13)], [y(1,11), y(1,13)],'color', 'g','linewidth', 3);
  % line([x(1,9), x(1,11)], [y(1,9), y(1,11)],'color', 'y','linewidth', 3);
  line([x(1,13), x(1,15)], [y(1,13), y(1,15)],'color', 'y','linewidth', 3);

hold off;
drawnow;
end

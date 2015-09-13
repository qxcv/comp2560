% display the detected pose sequence
function show_pose_sequence(data_path, frames, detected_pose_seq)
figno=1023; figure(figno);clf(figno); 

for n = 1:length(frames)   
    %video_showskeleton2(img{n}, detected_pose_seq(n,:));axis off;
    video_showskeleton2([data_path frames(n).name], detected_pose_seq(n,:));axis off;
    pause(1);
end
end

function video_showskeleton2(frame, boxes)
 parent = [0 1 2 3 4 5 1 7 8 9 10 1 12];
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
  line([x(1,2), x(1,4)], [y(1,2), y(1,4)],'color', 'r','linewidth', 3);
  line([x(1,4), x(1,6)], [y(1,4), y(1,6)],'color', 'b','linewidth', 3);
  line([x(1,7), x(1,9)], [y(1,7), y(1,9)],'color', 'g','linewidth', 3);
  line([x(1,9), x(1,11)], [y(1,9), y(1,11)],'color', 'y','linewidth', 3);

hold off;
drawnow;
end

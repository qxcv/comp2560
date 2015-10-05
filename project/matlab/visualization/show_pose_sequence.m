% display the detected pose sequence
function show_pose_sequence(data_path, frames, detected_pose_seq, parent, dest_dir)

if nargin < 5
    should_save = false;
else
    should_save = true;
    mkdir(dest_dir);
end

figno=1023; figure(figno); clf(figno); 

for n = 1:length(frames)
    video_showskeleton2([data_path frames(n).name], detected_pose_seq(n,:), parent);
    axis off;
    if should_save
        dest_path = fullfile(dest_dir, sprintf('frame-%i', n));
        fprintf('Saving to %s\n', dest_path);
        print(dest_path, '-dpng');
        pause(0.2);
    else
        pause(1);
    end
end
end

function video_showskeleton2(frame, boxes, parent)
 img = imread(frame);
 imshow(img); hold on; 
  
  for i = 1:length(parent)
    x1(:,i) = boxes(:,1+(i-1)*4);
    y1(:,i) = boxes(:,2+(i-1)*4);
    x2(:,i) = boxes(:,3+(i-1)*4);
    y2(:,i) = boxes(:,4+(i-1)*4);
  end
  x = (x1 + x2)/2;
  y = (y1 + y2)/2;  
  
  hold on;
  line([x(1,3), x(1,5)], [y(1,3), y(1,5)],'color', 'r','linewidth', 3);
  line([x(1,5), x(1,7)], [y(1,5), y(1,7)],'color', 'b','linewidth', 3);
  line([x(1,11), x(1,13)], [y(1,11), y(1,13)],'color', 'g','linewidth', 3);
  line([x(1,13), x(1,15)], [y(1,13), y(1,15)],'color', 'y','linewidth', 3);

hold off;
drawnow;
end

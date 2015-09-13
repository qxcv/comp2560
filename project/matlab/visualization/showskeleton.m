% displays a skeleton for the learned piw pose model. use the pts to
% display the poses given keypoints, and use boxes if box coordinates are
% available (e.g., pose boxes returned from loopy_detect_MM.m). np defines
% the limbs to display and visiblility is the occlusion flag. 
function showskeleton(img, pts, boxes, np, visibility)
 parent = [0 1 2 3 4 5 1 7 8 9  10 1  12];
 partcolor = {'g','g', 'y', 'm','r', 'g', 'c','m', 'r', 'b','g', 'y','m','g','r'};
 if nargin <=2
     boxes = [];
 end
 if nargin <= 3
     if isempty(np)
         np = 2:11; 
     end
     if isempty(visibility)
         visibility = ones(1,length(parent));
     end
 end
 
 imagesc(img); hold on; 
 if isempty(pts)
     if ~isempty(boxes)
      for i = 1:length(parent)
        x1(:,i) = boxes(:,1+(i-1)*4);
        y1(:,i) = boxes(:,2+(i-1)*4);
        x2(:,i) = boxes(:,3+(i-1)*4);
        y2(:,i) = boxes(:,4+(i-1)*4);
      end
      x = (x1 + x2)/2;
      y = (y1 + y2)/2;
      
      for n = 1:min(1000,size(boxes,1))
        for child = np%2:18%[2,3,11]
          %  pause;
          if visibility(child)              
              x1 = x(n,parent(child));
              y1 = y(n,parent(child));
              x2 = x(n,child);
              y2 = y(n,child);               
              line([x1 x2],[y1 y2],'color',partcolor{child},'linewidth',2);          
          end
        end
        %pause;
      end
     end
     hold off;
     drawnow;
 else     
     if max(size(pts)==length(parent))
         if size(pts,1)>size(pts,2), pts=pts';end
         pa = parent;
         for i=2:size(pts,2)          
             if visibility(i) %&& ~sum(pts(:,i)==-1) && ~sum(pts(:,pa(i))==-1)
                line([pts(1,i), pts(1,pa(i))], [pts(2,i), pts(2,pa(i))], 'color', partcolor{i}, 'linewidth', 2);
             end
         end
     else
        colors = {'r', 'b', 'y', 'g', 'm', 'w'};
        imshow(img);    
        hold on;
        for j=1:size(pts,2)
            if visibility(j)
                line(pts(1:2,j), pts(3:4,j), 'color',colors{j}, 'linewidth', 3);               
            end
        end
        hold off;                
     end
 end
end


% takes the start and end of limb positions and makes a color histogram descriptor
% for the entire limb rather than patches around the keypoint. numpts
% specifies how many points to be taken from the limb (at equidistant
% points).
function pts = get_pts_along_limb(img, yst, xst, yen, xen, ps, szx, szy, numpts) 
region = get_rect_coordinates([yst, xst], [yen, xen], ps,szx, szy);
% regions.pts = {}; regions = repmat(regions, [size(region,1), 1]); % every region is a limb location of a pose.

n = size(yst,1);
stmid = [(region(:,1)+region(:,2))/2, (region(:,5) + region(:,6))/2];
enmid = [(region(:,3)+region(:,4))/2, (region(:,7) + region(:,8))/2];
th = atan2(enmid(:,2)-stmid(:,2), enmid(:,1)-stmid(:,1)+eps);
costh = cos(th); sinth = sin(th);
%T = arrayfun(@(t) [cos(t) sin(t); -sin(t) cos(t)], th, 'uniformoutput', false);
limblen=zeros(n,1);
for p=1:n
    limblen(p) = norm(stmid(p,:)-enmid(p,:),'fro')/(numpts+eps);
end
%pts = arrayfun(@(p) stmid(p,:) + (0:numpts-1).*[limblen(p), 0]*[cos(th(p)) sin(th(p)); -sin(th(p)) cos(th(p))], 1:n, 'uniformoutput', false);
pts = zeros(n,2*numpts); %one = ones(1,numpts);
for p=1:n
    R = (0:(numpts-1))*limblen(p);
    pts(p,1:2:2*numpts) = stmid(p,1) + R*costh(p); 
    pts(p,2:2:2*numpts) = stmid(p,2) + R*sinth(p);    
    %pts(p,:) = vec(stmid(p,:)'*one + [R*costh(p); R*sinth(p)])'; % this is basically stmid + [cos(Th), -sin(th); sin(th) cos(th)]*(0:numpts-1)*[limblen;0];
end
%pts = pts';
%pts = arrayfun(@(p) vec(repmat(stmid(p,:)',[1,numpts]) + [cos(th(p)), -sin(th(p)); sin(th(p)), cos(th(p))]*[(0:(numpts-1))*limblen(p); Z])', 1:n, 'uniformoutput', false);


% regions = arrayfun(@(p) get_pts_along_principal_axis(img, region(p,1:4), region(p,5:8), ps, numpts), 1:n, 'uniformoutput', false);
% regions = cell2mat(regions);

% for i=1:size(region,1)    
%     regions(i).pts = get_pts_along_principal_axis(img, region(i,1:4), region(i,5:8), ps, numpts);    
% end
end

% rather than using roi poly which is slow, we will extract patches from a
% line joining the principal axis of the roi.
function pts = get_pts_along_principal_axis(img, st, en, ps, numpts)
stmid=[(st(1)+st(2))/2, (en(1)+en(2))/2]';
enmid=[(st(3)+st(3))/2, (en(3)+en(4))/2]';
th = atan2((enmid(2)-stmid(2)), (enmid(1)-stmid(1)+eps));

T = [cos(th), -sin(th); sin(th), cos(th)];
% imshow(img); hold on;  pat={}; 
p = 0; n = 0; pts=cell(numpts,1);
limblen = norm(stmid-enmid,'fro'); ps = limblen/(numpts+eps);
%pts = stmid + (0:n-1).*T*[ps;0];
%pts = pts';
% 
for n=1:numpts
    p = stmid + (n-1)*T*[ps;0];
    pts{n} = p;
end
%{
while norm(p-enmid)>ps/2    
    p = stmid + n*T*[ps;0];
    %plot(p(1), p(2), 'rx');        
    n=n+1;    
    pts{n} = p;     
end
%}
end%%

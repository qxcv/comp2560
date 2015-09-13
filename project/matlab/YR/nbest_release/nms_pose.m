function pick = nms_pose(boxes,overlap,partI)

if isempty(boxes)
		pick = [];
		return;
end

if nargin < 2
    overlap = 0.5;
end
if nargin < 3
    partI = 1:floor(size(boxes,2)/4);
end

x1 = zeros(size(boxes,1),length(partI));
y1 = zeros(size(boxes,1),length(partI));
x2 = zeros(size(boxes,1),length(partI));
y2 = zeros(size(boxes,1),length(partI));
area = zeros(size(boxes,1),length(partI));
%%%	for p = 1:numparts
for p = 1:length(partI)
		pp = partI(p);
		x1(:,p) = boxes(:,1+(pp-1)*4);
		y1(:,p) = boxes(:,2+(pp-1)*4);
		x2(:,p) = boxes(:,3+(pp-1)*4);
		y2(:,p) = boxes(:,4+(pp-1)*4);
		area(:,p) = (x2(:,p)-x1(:,p)+1) .* (y2(:,p)-y1(:,p)+1);
end

s = boxes(:,end);
[vals, I] = sort(s);
pick = [];
while ~isempty(I)

	last = length(I);
	i = I(last);
	pick = [pick ; i];
	
	suppress = 1:last-1;
	j = I(suppress);

	for p = 1:length(partI)
		if isempty(j)
			break;
		end

		xx1 = max(x1(j,p),x1(i,p));
		yy1 = max(y1(j,p),y1(i,p));
		xx2 = min(x2(j,p),x2(i,p));
		yy2 = min(y2(j,p),y2(i,p));

		w = xx2-xx1+1;
		h = yy2-yy1+1;

		ov = find(w>0 & h>0);
		if isempty(ov)
			suppress = [];
			break;
		end

		inters = w(ov).*h(ov);
		o = inters./(area(i) + area(j(ov)) -inters);
		
		ind2 = find(o>overlap);
		ind = ov(ind2);

		suppress = suppress(ind); 
		j =  I(suppress);
	end

	suppress = [suppress(:) ; last];

	I(suppress) = [];

end




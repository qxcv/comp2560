% returns the coordinates of a rectangle with the main diagonal as the
% coordinates given by start and end.
function region = get_rect_coordinates(startcor, endcor, patch_size, szx, szy)
x1 = startcor(:,2); y1=startcor(:,1); x2 = endcor(:,2); y2=endcor(:,1);
tantheta = (y2-y1)./(x2-x1+eps); theta=atan(tantheta);
w = patch_size;
A = [y1+ (w/2).*cos(theta), x1 + (w/2).*sin(theta)]; A((A(:,1)>szx))=szx; A((A(:,2)>szy))=szy;
B = [y1- (w/2).*cos(theta), x1 - (w/2).*sin(theta)]; B((B(:,1)<1))=1; B((B(:,2)<1))=1;
C = [y2- (w/2).*cos(theta), x2 - (w/2).*sin(theta)]; C((C(:,1)<1))=1; C((C(:,2)<1))=1;
D = [y2+ (w/2).*cos(theta), x2 + (w/2).*sin(theta)]; D((D(:,1)>szx))=szx; D((D(:,2)>szy))=szy;

%hold on; 
%line([A(1), B(1)], [A(2), B(2)]); line([B(1), C(1)], [B(2), C(2)]);
%line([C(1), D(1)], [C(2), D(2)]); line([D(1), A(1)], [D(2), A(2)]);
% region = [A(1), A(2), B(1), B(2), C(1), C(2), D(1), D(2);
%           B(1), B(2), C(1), C(2), D(1), D(2), A(1), A(2)];
region = [A(:,1), B(:,1), C(:,1), D(:,1), A(:,2), B(:,2), C(:,2), D(:,2)];
end

% 
% function region = get_rect_coordinates(startcor, endcor, patch_size, szx, szy)
% x1 = startcor(2); y1=startcor(1); x2 = endcor(2); y2=endcor(1);
% tantheta = (y2-y1)/(x2-x1+eps); theta=atan(tantheta);
% w = patch_size;
% A = [y1+ (w/2)*cos(theta), x1 + (w/2)*sin(theta)]; A((A(:,1)>szx))=szx; A((A(:,2)>szy))=szy;
% B = [y1- (w/2)*cos(theta), x1 - (w/2)*sin(theta)]; B((B(:,1)<1))=1; B((B(:,2)<1))=1;
% C = [y2- (w/2)*cos(theta), x2 - (w/2)*sin(theta)]; C((C(:,1)<1))=1; C((C(:,2)<1))=1;
% D = [y2+ (w/2)*cos(theta), x2 + (w/2)*sin(theta)]; D((D(:,1)>szx))=szx; D((D(:,2)>szy))=szy;
% 
% %hold on; 
% %line([A(1), B(1)], [A(2), B(2)]); line([B(1), C(1)], [B(2), C(2)]);
% %line([C(1), D(1)], [C(2), D(2)]); line([D(1), A(1)], [D(2), A(2)]);
% % region = [A(1), A(2), B(1), B(2), C(1), C(2), D(1), D(2);
% %           B(1), B(2), C(1), C(2), D(1), D(2), A(1), A(2)];
% region = [A(1), B(1), C(1), D(1);
%           A(2), B(2), C(2), D(2)];
% end%%

function hf= get_motion_from_flow(pt, opticalflow, display_poses)
addpath('/home/lear/cherian/projects/pose/MyCode/sequence/shortest-path/');
ps = 15;
u = opticalflow.u; v=opticalflow.v;
n = size(pt,1);

%hf = zeros(n, 2);
str = max(1,pt(:,1)-ps)+1; enr = min(size(u,2), pt(:,1)+ps);
stc = max(1,pt(:,2)-ps)+1; enc = min(size(u,1), pt(:,2)+ps);
% patch_area=zeros(n,1);
% for i=1:n
%     patch_area(i) = (enr(p)-str(p)+1)*(enc(p)-stc(p)+1);
% end


% idxrow_range = arrayfun(@(p) round(stc(p):enc(p)), 1:n,'uniformoutput', false);
% idxcol_range = arrayfun(@(p) round(str(p):enr(p)), 1:n, 'uniformoutput', false);
stc=round(stc); enc=round(enc); str=round(str); enr=round(enr);
hf = zeros(n,2);
for p=1:n
    idxrow = stc(p):enc(p); idxcol = str(p):enr(p);
    %hf(p,:) = [mymean(u(idxrow, idxcol)), mymean(v(idxrow, idxcol))];    
    if numel(idxrow)~=0 && numel(idxcol)~=0
        hf(p,1) = mymax(u(idxrow, idxcol));        
        hf(p,2) = mymax(v(idxrow, idxcol));
    else
        hf(p,:) = [0,0];
    end
end

% hf(:,1) = arrayfun(@(p) mean(mean(u(round(stc(p):enc(p)), round(str(p):enr(p))))), 1:n);
% hf(:,2) = arrayfun(@(p) mean(mean(v(round(stc(p):enc(p)), round(str(p):enr(p))))), 1:n);

% for i=1:size(pt,1)       
%     %U = u(stc(i):enc(i), str(i):enr(i));
%     %V = v(stc(i):enc(i), str(i):enr(i));
%     %idxrow = stc(i):enc(i); idxcol = str(i):enr(i);
%     
%     %[~,b] = max(abs(vec(U))); hf(i,1) = U(b);    
%     %hf(i,1) = mean(U(:)); %U(b); %(U(:)); 
%     hf(i,1) = sum(sum(u(stc(i):enc(i), str(i):enr(i))));
%     hf(i,2) = sum(sum(v(stc(i):enc(i), str(i):enr(i))));
%     
%     %[~,b] = max(abs(vec(V))); hf(i,2) = V(b); 
%     %hf(i,2) = mean(V(:)); %V(b);%mean(V(:)); %V(b);    
% %     if display_poses == 1
% %         line([pt(i,1), pt(i,1)+hf(i,1)], [pt(i,2), pt(i,2)+hf(i,2)], 'color', 'g', 'linewidth', 3); 
% %     end
% end

end

function m=mymedian(u)
m = median(u(:));
end

%{
function m=mymax(u)
% if isempty(u), 
%     m=0;
% else
    P = abs(u(:));
    [~,idx] = max(abs(P),[],1); 
    m = u(idx);
% end
end
%}
function m=mymean(u)
m = sum(u(:))/numel(u);
end

function m=mysum(u)
m = sum(u(:));
end%%

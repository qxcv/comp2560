% get a patch of size pat_size from img1 around point x,y
function colordesc = get_patch_descr(img, x,y, ps, szx, szy)
N = size(y,1);
%bins = 0:0.01:1; nbins = length(bins); 
nbins = 8; colordesc = zeros(512, N);
for i=1:N
    try
         pat = img(x(i)-ps+1:x(i)+ps, y(i)-ps+1:y(i)+ps,:);         
    catch    
        pat = zeros(2*ps,2*ps,3);    
        mypat = img(max(1, round(x(i)-ps+1)):min(szx, round(x(i)+ps)), max(1,round(y(i)-ps+1)):min(szy, round(y(i)+ps)),:);
        pat(1:size(mypat,1), 1:size(mypat,2),:) = mypat;  
    end
     
    colordesc(:,i) = vec(pat);%colorhist(reshape(pat,[size(pat,1)*size(pat,2),3]), nbins); % nbins = 8.
    %colordesc(:, i) = [histc(vec(pat(:,:,1)), bins); histc(vec(pat(:,:,2)), bins); histc(vec(pat(:,:,3)), bins)];    
end
%colordesc = bsxfun(@rdivide, colordesc, sum(colordesc,1));
end

% function h = color_hist(patch, bins)
% 
% patch = reshape(patch, [size(patch,1)*size(patch,2), 3]);
% for i=1:3
%     h((i-1)*bin_size+1:i*bin_size) = histc(patch(:,i), bins);
% end
% h = h/sum(h);
% end%%

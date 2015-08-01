% assumes the images is color. takes the image and the number of bins in
% each axis. We assume the image is nx2 where the rows represent the pixels
% and the cols are the color channels.

function H = colorhist(img, nbins)
    persistent bins;
    if isempty(bins)
        if isa(img, 'double')% || max(img(:))>1
            bins = linspace(1/nbins,1, nbins);
        else
            bins = linspace(0, 255, nbins);
        end
    end
    img = reshape(img,[size(img,1)*size(img,2),size(img,3)]);
    %H = [histc(img(:,1), bins); histc(img(:,2), bins); histc(img(:,3), bins)]';
    H = histc(img,bins); H = H(:)';
    H = H/(sum(H)+eps);   
end

%{
function H = colorhist(img, nbins)
    persistent bins;
    if isempty(bins)
        if isa(img, 'double')
            bins = linspace(1/nbins,1, nbins);
        else
            bins = linspace(0, 255, nbins);
        end
    end
    
    %H = zeros(nbins, nbins, nbins); % nbins for each color channel.    
    %H = zeros(nbins*nbins*nbins,1);
    img = reshape(img, [size(img,1)*size(img,2), 3]);         
    B = ones(size(img,1),1)*bins; %B=repmat(bins, [size(img,1),1]); 
    idx = zeros(size(img,1),3);
    for i=1:3        
        I = double(img(:,i))*ones(1,size(bins,2)); %I = repmat(img(:,i),[1,size(bins,2)]);
        S = sum(B>=I,2);
        idx(:,i) = nbins+1 - S;
    end    
    Hidx = (idx(:,1)-1)*64 + (idx(:,2)-1)*8 + idx(:,3);   
    H = histc(Hidx, 1:1:512);        
    H = H/(sum(H));    
%     if H(1)>0.9 % basically all colors are in the first bin which is black!
%         %disp('zero!!');
%         H(:)=0/0;
%     end
end
%}
%     ridx = arrayfun(@(x) find(bins>=x, 1, 'first'), img(:,1));
%     gidx = arrayfun(@(x) find(bins>=x, 1, 'first'), img(:,2));
%     bidx = arrayfun(@(x) find(bins>=x, 1, 'first'), img(:,3));
%     for i=1:length(ridx)
%         H(ridx(i),gidx(i),bidx(i)) = H(ridx(i),gidx(i),bidx(i)) + 1;
%     end
%     H = zeros(nbins, nbins, nbins);
%     for i=1:size(img,1)        
%         ridx=find(bins>=img(i,1), 1, 'first'); gidx = find(bins>=img(i,2), 1, 'first'); bidx = find(bins>=img(i,3),1, 'first');
%         H(ridx, gidx, bidx) = H(ridx, gidx, bidx) + 1;        
%     end



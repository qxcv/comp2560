function skelclr = get_patch_descr_Ex(skelpts, img, keyjoints, numpts_along_limb, displayposes)
    nbins=8;
    skelclr = cell(length(keyjoints), numpts_along_limb);
    persistent bins;
    if isempty(bins)
        if max(img(:))<=1 %isa(img, 'double')% ||
            bins = linspace(1/nbins,1, nbins);
        else
            bins = linspace(0, 255, nbins);
        end
    end
    
    for k=keyjoints
        for s=1:numpts_along_limb
            valid_range = (k-1)*numpts_along_limb*2 + [(s-1)*2+1 : 2*s];
            
            skelclr{k,s} = get_clrhist_from_img(skelpts(:,valid_range), img, bins, displayposes);
        end
    end
end

function hf= get_clrhist_from_img(pt, img, bins, display_poses)
    ps = 8; n = size(pt,1); histsize = 24;
    str = max(1,pt(:,1)-ps)+1; enr = min(size(img,2), pt(:,1)+ps);
    stc = max(1,pt(:,2)-ps)+1; enc = min(size(img,1), pt(:,2)+ps);

    stc=round(stc); enc=round(enc); str=round(str); enr=round(enr);    
    hf = zeros(histsize, n);
    if display_poses==1
        figure(1); imshow(img);
    end

    for p=1:n
        idxrow = stc(p):enc(p); idxcol = str(p):enr(p);                                          
        H = histc( img(idxrow, idxcol,:), bins, 1); H = sum(H,2);
        hf(:,p) = H(:);                
    end
    hf = bsxfun(@rdivide, hf, sum(hf,1)+eps);
    hf = hf';
end
%%

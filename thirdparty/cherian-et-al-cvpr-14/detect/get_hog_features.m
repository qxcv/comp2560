function feat=get_hog_features(imgex, sbin)        
    if size(imgex,3)==1,
        imgex=repmat(imgex,[1,1,3]);
    end
    feat = features(imgex(:,:,1:3), sbin); % hog
end


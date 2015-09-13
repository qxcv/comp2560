function clusters = cluster_rp(imdata, tsize)
% cluster relative position
conf = global_conf();
pa = conf.pa;
K = conf.K;
R = 80;    % do R times clustering
% cluster pairwise positions
nbh_IDs = get_IDs(pa, K);
p_no = numel(pa);

clusters = cell(p_no,1);
for ii = 1:p_no
  clusters{ii} = cell(numel(nbh_IDs{ii}),1);    % mean relative positions, and variance (unused)
end

rng(0);     % to reproduce results
for p = 1:p_no
  for n = 1:numel(nbh_IDs{p})
    % do clustering for each pair independently
    cur = p;
    nbh = nbh_IDs{p}(n);
    X = zeros(numel(imdata), 2);
    invalid = false(numel(imdata), 1); % if there is no invalid field, all are valid.
    for ii = 1:numel(imdata)
      % current joint - neighbor joint
      X(ii,:) = ...
        norm_rp(imdata(ii), cur,  nbh, tsize);
      if isfield(imdata, 'invalid')
        invalid(ii) = imdata(ii).invalid(cur) || imdata(ii).invalid(nbh);
      end
    end
    valid_idx = find(~invalid);
    mean_X = mean(X(valid_idx,:),1);
    normX = bsxfun(@minus, X(valid_idx,:), mean_X);
    
    gInd = cell(1,R);
    cen  = cell(1,R);
    sumdist = zeros(1,R);
    
    parfor trial = 1:R
      [gInd{trial}, cen{trial}, sumdist(trial)] = k_means(normX, K);
    end
    % take the smallest distance one
    [minSum, ind] = min(sumdist);
    minSum = minSum / numel(imdata) / 2;
    
    fprintf(' %d->%d: minSum=%.5f, K = %d\n', cur, nbh, minSum, K);
    if 0
      color = {'g','y','r','m','b','c','k'};
      figure;
      hold on;
      title(sprintf(' %d->%d: minSum=%.5f, K = %d\n', cur, nbh, minSum, K));
      for k = 1:K
        imgid = false(numel(imdata), 1);
        imgid(valid_idx(gInd{ind(1)}==k)) = true;
        LX = X(imgid, :);
        plot(LX(:,1), LX(:,2), ['.', color{mod(k,numel(color))+1}]);
        % draw center
        cent = cen{ind(1)}(k,:);
        plot(cent(1)+mean_X(1), cent(2)+mean_X(2), ['.', color{mod(k+1,numel(color)) + 1}], 'markersize', 15);
      end
      hold off;
    end
    
    for k = 1:K
      imgid = false(numel(imdata), 1);
      imgid(valid_idx(gInd{ind(1)}==k)) = true;
      
      LX = X(imgid, :);
      center = mean(LX,1);
      sigma = sqrt(var(LX, 0, 1));
      
      clusters{cur}{n}(k).imgid = imgid;
      clusters{cur}{n}(k).center = center;
      clusters{cur}{n}(k).sigma = sigma;
    end
  end
end

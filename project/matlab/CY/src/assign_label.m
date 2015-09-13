function labels = assign_label(imdata, clusters, pa, tsize, K, is_check)
if ~exist('is_check', 'var')
  is_check = false;
end
% add mix field to imgs
p_no = numel(pa);
labels = struct( 'mix_id', cell(numel(imdata), 1), ...
  'global_id', cell(numel(imdata), 1), ...
  'near', cell(numel(imdata), 1), ...
  'invalid', cell(numel(imdata), 1) );
[nbh_IDs, global_IDs] = get_IDs(pa, K);

parfor ii = 1:length(imdata)
  labels(ii).mix_id = cell(p_no, 1);
  labels(ii).near = cell(p_no, 1);
  labels(ii).invalid = false(p_no, 1);
  for p = 1:p_no
    nbh_N = numel(clusters{p});
    labels(ii).mix_id{p} = zeros(nbh_N, 1, 'int32');
    labels(ii).near{p} = cell(nbh_N, 1);
    invalid = false;
    for n = 1:nbh_N
      % find nearest
      nbh_idx = nbh_IDs{p}(n);
      if ( isfield(imdata, 'invalid') && (imdata(ii).invalid(p) || imdata(ii).invalid(nbh_idx)) )
        invalid = true;
      end
      cluster_num = numel(clusters{p}{n});
      centers = zeros(cluster_num, 2);
      for k = 1:cluster_num
        centers(k,:) = clusters{p}{n}(k).center;
        % sigmas(k,:) = clusters{p}{n}(k).sigma;
      end
      rp = norm_rp(imdata(ii), p, nbh_idx, tsize);
      
      dists = bsxfun(@minus, centers, rp);
      dists = sqrt(sum(dists .^ 2, 2));
      
      [~,id] = min(dists,[],1);
      % for debug
      if (is_check && ~invalid)
        is_imgid = clusters{p}{n}(id).imgid;
        assert(is_imgid(ii));
      end
      labels(ii).mix_id{p}(n) = int32(id);
      labels(ii).near{p}{n} = dists < 3 * dists(id);
    end
    % invalid
    labels(ii).invalid(p) = invalid;
  end
  labels(ii).global_id = int32(mix2global(labels(ii).mix_id, global_IDs));
end

function global_id = mix2global(mix_id, global_IDs)
p_no = numel(mix_id);
global_id = zeros(p_no, 1);
for p = 1:p_no
  mixs = ones(3, 1);
  for ii = 1:numel(mix_id{p})
    mixs(ii) = mix_id{p}(ii);
  end
  global_id(p) = global_IDs{p}(mixs(1),mixs(2),mixs(3));
end

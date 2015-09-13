function boxes = detect_fast(iminfo, model, thresh, param)
% boxes = detect(im, model, thresh)
% Detect objects in input using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part

% Compute the feature pyramid and prepare filter
INF = 1e10;

useGpu = param.useGpu;
at_least_one = param.at_least_one;

test_with_detection = param.test_with_detection;
if test_with_detection && isempty(iminfo.constr_bbx)
  boxes = [];
  return;
end

im = imreadx(iminfo);
if test_with_detection
  % crop and scale image using detection information to accelerate pose
  % estimation
  full_bbx = round(iminfo.full_bbx); %  + [-psize, +psize]);
  dim1 = size(im, 1); dim2 = size(im, 2);
  full_bbx = [max(1, full_bbx(1)), max(1, full_bbx(2)), ...
    min(dim2, full_bbx(3)), min(dim1, full_bbx(4))];
  im = subarray(im, full_bbx(2), full_bbx(4), full_bbx(1), full_bbx(3), 0);
  iminfo.constr_bbx = iminfo.constr_bbx - [full_bbx(1), full_bbx(2), full_bbx(1), full_bbx(2)] + 1;
else
  full_bbx = [];
end
[pyra, unary_map, idpr_map] = imCNNdet(im,model,useGpu,1,param.impyra_fun);

% Cache various statistics derived from model
[components,apps] = modelcomponents(model);

levels = 1:length(pyra);
boxes = cell(numel(levels), 1);

% Iterate over scales and components,
parfor level = levels,
  sizs = pyra(level).sizs;
  for c  = 1:length(model.components),
    parts    = components{c};
    p_no = length(parts);
    % Local scores
    for p = 1:p_no
      % assign each deformation scores
      parts(p).defMap = idpr_map{level}{p};
      parts(p).appMap = unary_map{level}{p};
      
      f = parts(p).appid;
      parts(p).score = parts(p).appMap * apps{f};
      parts(p).level = level;
    end
    % constraint head position if we test_with_detection
    if test_with_detection
      constr_bbx = iminfo.constr_bbx;
      bbx2map = (constr_bbx - 1) ./ pyra(level).scale + 1;
      bbx2map = floor(bbx2map);
      % constr_bbx should be guaranteed to lie inside image
%       if ~(bbx2map(1) >= 1 && bbx2map(2) >= 1 ...
%           && bbx2map(3) <= sizs(2) && bbx2map(4) <= sizs(1))
%         fprintf('detect_fast: Warning: bbx2map larger than map: %s\n', iminfo.im);
%       end
      bbx2map([1,2]) = max(1, bbx2map([1,2]));
      bbx2map([3,4]) = min([sizs(2), sizs(1)], bbx2map([3,4]));
      
      invalid_map = true(sizs(1), sizs(2));
      invalid_map(bbx2map(2):bbx2map(4), bbx2map(1):bbx2map(3)) = false;
      for cpi = 1:numel(param.constrainted_pids)
        parts(param.constrainted_pids(cpi)).score(invalid_map) = -INF;
      end
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for p = p_no:-1:2,
      child = parts(p);
      par = parts(p).parent;
      parent = parts(par);
      cbid = find(child.nbh_IDs == parent.pid);
      pbid = find(parent.nbh_IDs == child.pid);
      
      [msg,parts(p).Ix,parts(p).Iy,parts(p).Im{cbid},parts(par).Im{pbid}] ...
        = passmsg(child, parent, cbid, pbid);
      parts(par).score = parts(par).score + msg;
    end
    
    % Add bias to root score
    parts(1).score = parts(1).score + parts(1).b;
    rscore = parts(1).score;
    
    % modify thresh to keep at least one response at each level
    if at_least_one
      lev_thre = min(thresh,max(rscore(:)));
    else
      lev_thre = thresh;
    end
    % Walk back down tree following pointers
    [Y,X] = find(rscore >= lev_thre);
    if ~isempty(X)
      I   = (X-1)*size(rscore,1) + Y;
      box = backtrack(X,Y,parts,pyra(level));
      numx = length(X); numparts = length(parts);
      if test_with_detection
        box = bsxfun(@plus, box, [full_bbx(1), full_bbx(2), full_bbx(1), full_bbx(2)]);
        box = box - 1;
      end
      box = reshape(box,numx,4*numparts);

      boxes{level} = [box repmat(c,length(I),1) rscore(I)];
    end
  end
end
boxes = cat(1, boxes{:});

% Backtrack through DP msgs to collect ptrs to part locations
function box = backtrack(x,y,parts,pyra)
numx     = length(x);
numparts = length(parts);

xptr = zeros(numx,numparts);
yptr = zeros(numx,numparts);
box  = zeros(numx,4,numparts);

for k = 1:numparts,
  p   = parts(k);
  if k == 1,
    xptr(:,k) = x;
    yptr(:,k) = y;
  else
    % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
    par = p.parent;
    [h,w,foo] = size(p.Ix);
    I   = (xptr(:,par)-1)*h + yptr(:,par);
    xptr(:,k) = p.Ix(I);
    yptr(:,k) = p.Iy(I);
    
  end
  scale = pyra.scale;
  x1 = (xptr(:,k) - 1 - double(pyra.padx))*scale+1;
  y1 = (yptr(:,k) - 1 - double(pyra.pady))*scale+1;
  x2 = x1 + double(p.sizx)*scale - 1;
  y2 = y1 + double(p.sizy)*scale - 1;
  box(:,:,k) = [x1 y1 x2 y2];
end


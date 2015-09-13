function boxes = test_model(note,model,test)
% boxes = testmodel(name,model,test,suffix)
% Returns candidate bounding boxes after non-maximum suppression

conf = global_conf();

cachedir = conf.cachedir;
par.impyra_fun = conf.impyra_fun;
par.useGpu = conf.useGpu;
par.device_id = conf.device_id;
par.at_least_one = conf.at_least_one;
par.test_with_detection = conf.test_with_detection;
if par.test_with_detection
  par.constrainted_pids = conf.constrainted_pids;
end

try
  boxes = parload([cachedir note '_raw_boxes'], 'boxes');
catch
  boxes = cell(1,length(test));
  %     parfor i = 1:length(test)
  for i = 1:length(test)
    fprintf([note ': testing: %d/%d\n'],i,length(test));
    box = detect_fast(test(i),model,model.thresh,par);
    boxes{i} = nms_pose(box,0.3);
  end
  parsave([cachedir note '_raw_boxes'], boxes);
end

for i = 1:length(test)
  if ~isempty(boxes{i})
    % only keep the highest scoring estimation for evaluation.
    boxes{i} = boxes{i}(1,:);
  end
  % visualization
  if 0
    im = imreadx(test(i));
    if ~isempty(boxes{i})
      showskeletons(im, boxes{i}(1,:), conf.pa);
      pause;
    end
  end
end

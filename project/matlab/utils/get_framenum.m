% some function to return a unique id for a frame. unique in the dataset.
function frameid = get_framenum(imname)
   str = strtok(strrep(imname, '-', ''), '.');
   str = str(str<'A'); 
   frameid = str2double(str);
end
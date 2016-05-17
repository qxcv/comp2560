function result = hasfield(struct, field_name)
%HASFIELD Check whether a cell/struct array has a given field
names = fieldnames(struct);
result = false;
for i=1:length(names)
    if strcmp(field_name, names(i))
        result = true;
        break
    end
end
end


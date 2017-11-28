f = figure();
h = uicontrol('Parent',f,'Style','slider','Position',[81,54,419,23],...
              'value',0, 'min',0, 'max',5);
addlistener(h, 'Value', 'PostSet', @updateValue);


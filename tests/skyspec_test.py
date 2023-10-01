from simms.skymodel.skydef import Line, Cont
from simms.config_spec import validate

line = Line(1.3, 4, 10.2)
validate(line)
cont = Cont(5.2, 10.6, [5.1,2.2,3.4,1])
validate(cont)
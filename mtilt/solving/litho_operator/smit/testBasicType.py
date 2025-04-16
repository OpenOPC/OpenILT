#!/usr/bin/python3
import gwxopc_md as md
int_a = md.basictype.Int(0) #type:BasicInt
int_a.setValue(3)
print(int_a, int_a.getValue())
bool_b = md.basictype.Bool(1) #type:BasicBool
print(bool_b, bool_b.getValue())
str_c = md.basictype.Str("basic string type: ")
str_d = md.basictype.Str(str_c.getValue()+"hello world")
print(str_d, str_d.getValue())

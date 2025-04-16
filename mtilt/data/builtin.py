from .benchmarks import register_iccad2013_designs


_PREDEFINED_SPLIT_DESIGN = {
    "iccad2013": "./benchmark/ICCAD2013",
}



def register_all_designs():
    for name, path in _PREDEFINED_SPLIT_DESIGN.items():
        register_iccad2013_designs(name, path)


register_all_designs()
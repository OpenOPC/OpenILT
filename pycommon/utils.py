def parseConfig(filename): 
    with open(filename, "r") as fin: 
        lines = fin.readlines()
    results = {}
    for line in lines: 
        splited = line.strip().split()
        if len(splited) >= 2: 
            key = splited[0]
            value = splited[1]
            results[key] = value
    return results
        
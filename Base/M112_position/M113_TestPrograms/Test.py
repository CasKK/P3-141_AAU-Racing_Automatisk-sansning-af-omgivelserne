inputA =    [[-350.13, 50.90, "blue"],
            [-350.34, 100.90, "blue"],
            [-350.56, 200.90, "blue"],
            [-350.13, 250.90, "blue"],
            [-350.34, 300.90, "blue"],
            [-350.56, 400.90, "blue"]]

print(inputA)

def someFunction(localInputA):
    for i, vector in enumerate(localInputA):
        result = vector[0] + 350
        localInputA[i] += [result]
    return localInputA

someVar = someFunction(inputA)

print(f"\n{inputA}")
print(f"\n{someVar}")
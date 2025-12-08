# Træningsdata
class_A = [[12.0, 0.85, 200],
           [15.0, 0.90, 180],
           [14.0, 0.88, 190]]

class_B = [[30.0, 0.40, 50],
           [28.0, 0.35, 60],
           [32.0, 0.45, 55]]

# Beregn mean og varians
def mean_var(data):
    n = len(data)
    features = len(data[0])
    means, vars_ = [], []
    for j in range(features):
        col = [row[j] for row in data]
        mu = sum(col) / n
        var = sum((x - mu)**2 for x in col) / n
        means.append(mu)
        vars_.append(var)
    return means, vars_

mu_A, var_A = mean_var(class_A)
mu_B, var_B = mean_var(class_B)

# Gem i txt (kun mean og var)
with open("model.txt", "w") as f:
    f.write("A_mean " + " ".join(map(str, mu_A)) + "\n")
    f.write("A_var " + " ".join(map(str, var_A)) + "\n")
    f.write("B_mean " + " ".join(map(str, mu_B)) + "\n")
    f.write("B_var " + " ".join(map(str, var_B)) + "\n")

print("Model gemt i model.txt")
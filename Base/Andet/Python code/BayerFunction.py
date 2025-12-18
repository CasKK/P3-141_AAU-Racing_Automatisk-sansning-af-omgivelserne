import math

# Læs model
def load_model(filename="model.txt"):
    model = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            values = [float(v) for v in parts[1:]]
            model[key] = values
    return model

# Log af Gaussisk tæthed
def gaussian_log_prob(x, mu, var):
    return -0.5*math.log(2*math.pi*var) - ((x-mu)**2)/(2*var)

# Beregn log-score
def class_log_score(x, mu, var, prior):
    log_probs = [gaussian_log_prob(xi, mui, vari) for xi, mui, vari in zip(x, mu, var)]
    return sum(log_probs) + math.log(prior)

# Indlæs model
model = load_model("model.txt")

# Ny blob
new_blob = [20.0, 0.70, 150]

# 👉 Her definerer du priors selv
prior_A = 0.5
prior_B = 0.5

# Beregn log-scores
logA = class_log_score(new_blob, model["A_mean"], model["A_var"], prior_A)
logB = class_log_score(new_blob, model["B_mean"], model["B_var"], prior_B)

# Konverter til sandsynligheder
pA = math.exp(logA)
pB = math.exp(logB)
total = pA + pB
prob_A = pA / total
prob_B = pB / total

print("Sandsynlighed for A:", prob_A)
print("Sandsynlighed for B:", prob_B)
print("Ny blob klassificeres som:", "A" if prob_A > prob_B else "B")

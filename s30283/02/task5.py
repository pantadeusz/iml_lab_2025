def bayes_medical_test(prior, sensitivity, specificity):
    # p(Choroba|Test+)
    p_test_given_disease = sensitivity          # p(E|H)
    p_test_given_no_disease = 1 - specificity   # p(E|~H)
    p_disease = prior                           # p(H)
    p_no_disease = 1 - prior                    # p(~H)
    
    # p(H|E) = p(H)p(E|H) / (p(H)p(E|H) + p(~H)p(E|~H))
    
    numerator = p_test_given_disease * p_disease
    denominator = numerator + p_test_given_no_disease * p_no_disease
    posterior = numerator / denominator
    return posterior

# Przykład z wykładu
post = bayes_medical_test(0.01, 0.99, 0.8)
print(f'P(Choroba|Test+) = {post:.3f}')

def bayes_multiple_tests(prior, sensitivity, specificity, results):
    p_disease = prior
    for result in results:
        if result:
            p_test_given_disease = sensitivity
            p_test_given_no_disease = 1 - specificity
        else:
            p_test_given_disease = 1 - sensitivity
            p_test_given_no_disease = specificity
        numerator = p_test_given_disease * p_disease
        denominator = numerator + p_test_given_no_disease * (1 - p_disease)
        # posterior becomes new prior
        p_disease = numerator / denominator
    return p_disease

print(bayes_multiple_tests(0.01, 0.99, 0.8, [1, 1, 1, 0]))
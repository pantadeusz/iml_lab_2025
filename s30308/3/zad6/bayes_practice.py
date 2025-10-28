"""
Notatki z zadania
------------------------------------------------------------------------------------------------------------------------

Czułość (ang. sensitivity, recall) - mówi, jak dobrze model wykrywa przypadki prawdziwie pozytywne.
Przykład:
Jeżeli model wykrywa 90 z 100 chorych → czułość = 90%.

------------------------------------------------------------------------------------------------------------------------

Swoistość (ang. specificity) - mówi, jak model dobrze rozpoznaje przypadki prawdziwie negatywne
Przykład:
Jeżeli test poprawnie identyfikuje 95 z 100 zdrowych → swoistość = 95%.

------------------------------------------------------------------------------------------------------------------------

Anatomia twierdzenia Bayesa:
- sensitivity – P(Test+|Choroba).
- specificity – P(Test-|NieChoroba).
- numerator – część wzoru Bayesa odpowiadająca „ile pozytywnych wyników pochodzi od chorych”.
- denominator – całkowite prawdopodobieństwo wyniku pozytywnego (pozytywne od chorych + fałszywie pozytywne od zdrowych).
- posterior – wynik końcowy: P(Choroba|Test+).

Bayes działa kaskadowo – każde nowe dane aktualizują wcześniejsze przekonanie (prior).
To dokładnie odzwierciedla sposób,
w jaki lekarz lub system diagnostyczny aktualizuje ocenę prawdopodobieństwa choroby po kolejnych testach:
Im bardziej pewny wynik z poprzedniego testu (wysoki posterior),
tym większe prawdopodobieństwo, że kolejny pozytywny wynik jeszcze bardziej potwierdzi chorobę.

------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

def bayes_medical_test(prior, sensitivity, specificity):
    """
    Oblicza posterior P(Choroba|Test+) używając Twierdzenia Bayesa
    """
    p_test_given_disease = sensitivity
    p_test_given_no_disease = 1 - specificity
    p_disease = prior
    p_no_disease = 1 - prior

    numerator = p_test_given_disease * p_disease
    denominator = numerator + p_test_given_no_disease * p_no_disease
    posterior = numerator / denominator
    return posterior

# Eksperyment 1: Różne częstości choroby
def experiment_prior_variation():
    priors = [0.001, 0.01, 0.1]  # rzadka, średnia, częsta choroba
    sensitivity = 0.99
    specificity = 0.95

    print("=== Eksperyment 1: Zmiana częstości choroby ===")
    for p in priors:
        post = bayes_medical_test(p, sensitivity, specificity)
        print(f'Prior={p:.3f} -> P(Choroba|Test+)={post:.3f}')


# Eksperyment 2: Wpływ swoistości (fałszywie dodatnich)
def experiment_specificity_variation():
    priors = np.linspace(0.001, 0.1, 50)
    sensitivity = 0.99
    specificities = [0.8, 0.9, 0.95, 0.99]

    plt.figure(figsize=(8,5))
    for spec in specificities:
        posts = [bayes_medical_test(p, sensitivity, spec) for p in priors]
        plt.plot(priors, posts, label=f'Swoistość={spec}')
    plt.xlabel('Częstość choroby (prior)')
    plt.ylabel('P(Choroba | Test+)')
    plt.title('Wpływ swoistości (fałszywie dodatnich) przy różnych priors')
    plt.legend()
    plt.grid(True)
    plt.show()


# Eksperyment 3: Sekwencja kilku testów
def experiment_multiple_tests():
    prior = 0.01
    sensitivity = 0.99
    specificity = 0.95
    tests = 3  # liczba testów w sekwencji

    print("=== Eksperyment 3: Sekwencja testów ===")
    posterior = prior
    for i in range(1, tests+1):
        posterior = bayes_medical_test(posterior, sensitivity, specificity)
        print(f'Test {i} -> P(Choroba|Test+)={posterior:.3f}')



if __name__ == '__main__':
    experiment_prior_variation()
    experiment_specificity_variation()
    experiment_multiple_tests()
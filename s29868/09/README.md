## Podsumowanie

Wprowadzono kilka zmian w architekturze modelu:

* **Funkcja aktywacji:** Zdecydowano się na użycie **tanh** zamiast standardowego ReLU. Głównym powodem była chęć uniknięcia zerowania ujemnych wartości (co ma miejsce w ReLU), dzięki czemu sieć zachowuje pełniejszą informację o sygnale wejściowym i na wyjściu obraz nie jest czarnym.


* **Architektura:** Największą poprawę wydajności zaobserwowano po zastosowaniu **sieci konwolucyjnych (CNN)**. W porównaniu do innych testowanych podejść, CNN poradziły sobie z zadaniem zdecydowanie najlepiej.
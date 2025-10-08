# Laboratorium 1: Wprowadzenie do Pythona i narzędzi ML

**Cele:**

- Opanowanie podstawowych umiejętności w Pythonie
- Tworzenie/obsługa repozytorium Git
- Tworzenie środowiska wirtualnego (venv)
- Korzystanie z edytorów kodu (VSCode lub PyCharm)
- Ładowanie danych za pomocą pandas
- Wizualizacja danych za pomocą matplotlib

## Uwagi ogólne

- Wszystkie laboratoria będą wykonywane w osobnych podkatalogach w repozytorium Git. Każdy student tworzy swój własny podkatalog o nazwie w formacie `s<indeks>`, gdzie `<indeks>` to indywidualny numer indeksu studenta (np. `s12345`).
- Można używać LLM-ów (np. Copilot) do przeglądu kodu i wyjaśnień, ale **nie do generowania całego kodu**. Ma to sens edukacyjny - jeśli agent wygeneruje wszystko, nie nauczycie się niczego. Używajcie LLM-ów do debugowania, optymalizacji lub zrozumienia błędów, ale piszcie kod samodzielnie.
- Zachęcam do własnej inwencji i niewielkiego modyfikowania nazw, treści i przykładów.

## Krok 1: Fork repozytorium Git

Uwaga: Studenci nie mają uprawnień do zapisu w głównym repozytorium, więc należy wykonać fork na swoje konto GitHub.

1. Przejdź do repozytorium głównego na GitHub (np. https://github.com/pantadeusz/iml_lab_2025).

2. Zaloguj się na swoje konto GitHub (lub utwórz jeśli nie masz).

3. Kliknij przycisk "Fork" w prawym górnym rogu, aby utworzyć fork repozytorium na swoim koncie.

4. Skopiuj URL swojego forka (HTTPS lub SSH).

5. W terminalu, przejdź do katalogu gdzie chcesz mieć projekt i sklonuj swój fork:

   ```bash
   git clone <URL_twojego_forka>
   cd <nazwa-repo>
   ```

6. Utwórz swój podkatalog dla laboratorium 1 (zastąp `<indeks>` swoim numerem indeksu):

   ```bash
   mkdir -p s<indeks>/01
   ```

   Na przykład, jeśli Twój indeks to 12345, użyj `s12345/01`.

7. Utwórz gałąź dla swojego laboratorium (aby uniknąć konfliktów i ułatwić zarządzanie zmianami):

   ```bash
   git checkout -b lab1-s<indeks>
   ```

   Na przykład: `git checkout -b lab1-s12345`

## Krok 2: Przygotowanie środowiska

1. Python powinien być dostępny na pracowni. Zaczniemy z poziomu konsoli, bez IDE.
2. Otwórz Git-Bash (lub inny terminal dostępny w pracowni) i przejdź do katalogu projektu.
3. Utwórz środowisko wirtualne:

   ```bash
   python -m venv lab_env
   ```

4. Aktywuj środowisko:
   - W Git-Bash (na Windows): `source lab_env/Scripts/activate`
   - W standardowym terminalu Windows: `lab_env\Scripts\activate`
   - Linux/Mac: `source lab_env/bin/activate`
5. Przygotuj plik requirements.txt i zainstaluj wymagane biblioteki:

   ```raw
   pandas
   matplotlib
   jupyter
   ```

   ```bash
   pip install -r requirements.txt
   ```

## Krok 3: Konfiguracja edytora

1. Otwórz VSCode lub PyCharm.
2. Skonfiguruj interpreter Pythona na utworzone środowisko wirtualne (lab_env).
3. Upewnij się, że rozszerzenia dla Pythona są zainstalowane (w VSCode: Python extension).

## Krok 4: Podstawy Pythona (20 minut)

1. W swoim podkatalogu `s<indeks>/01` utwórz plik `basics.py`.
2. Napisz kod, który:
   - Definiuje zmienną z Twoim imieniem i nazwiskiem
   - Tworzy listę z 5 liczbami
   - Używa pętli for do wyświetlenia elementów listy
   - Definiuje funkcję, która zwraca sumę dwóch liczb
   - Wywołuje funkcję i wyświetla wynik
3. Uruchom kod: `python basics.py`

## Krok 5: Ładowanie danych z pandas (20 minut)

1. Utwórz plik `data_loading.py` w `s<indeks>/01`.
2. Znajdź jakiś ciekawy tekstowy dataset CSV online, zachęcam do sprawdzenia <https://www.kaggle.com/datasets>.

   Jako wstęp, zobacz jak działa (i czy działa) u Ciebie pandas:

   ```python
   import pandas as pd
   # Przykładowe dane
   data = {'kolumna1': [1, 2, 3], 'kolumna2': [4, 5, 6]}
   df = pd.DataFrame(data)
   print(df.head())
   ```

3. Załaduj dane do DataFrame i wyświetl pierwsze 5 wierszy.

## Krok 6: Wizualizacja z matplotlib

1. Utwórz plik `plotting.py` w `s<indeks>/01`.
2. Użyj danych z poprzedniego kroku do stworzenia wykresu. I znowu przykład, jak zrobić wykres:

   ```python
   import matplotlib.pyplot as plt
   # Przykład wykresu liniowego
   plt.plot(df['kolumna1'], df['kolumna2'])
   plt.title('Przykładowy wykres')
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.show()
   ```

3. Zapisz wykres do pliku PNG.

## Krok 7: Kwestie składni (20 minut)

1. W podkatalogu `s<indeks>/01` utwórz plik `syntax_issues.py`.
2. Napisz skrypt, który:
   - Przyjmuje argumenty z linii komend: ścieżka do pliku CSV, nazwa kolumny, minimalna wartość zakresu, maksymalna wartość zakresu.
   - Załaduje dane z pliku CSV za pomocą pandas.
   - Wyfiltruje dane z podanej kolumny w zadanym zakresie wartości.
   - Wypisze wartości z wyznaczonego zakresu.
   - Wygeneruje histogram dla tych wartości za pomocą matplotlib i zapisze wykres do pliku PNG (np. `histogram.png`).
3. Uruchom skrypt z przykładowymi argumentami, np.:

   ```bash
   python syntax_issues.py <ścieżka_do_pliku_csv> <nazwa_kolumny> <min_wartość> <max_wartość>
   ```

## Krok 8: Podsumowanie i commit

1. Dodaj wszystkie pliki do Gita:

   ```bash
   git add s<indeks>/01/
   git commit -m "Lab 1: Podstawy Pythona i narzędzi ML"
   git push origin lab1-s<indeks>
   ```

2. Sprawdź status repozytorium.

3. Utwórz Pull Request (PR) do głównego repozytorium:
   - Przejdź do swojego forka na GitHub.
   - Kliknij "Compare & pull request".
   - Upewnij się, że base repository to główne repo (np. pantadeusz/iml_lab_2025), a head to Twój fork i gałąź `lab1-s<indeks>`.
   - Dodaj tytuł i opis PR, np. "Rozwiązanie lab 1 - s<indeks>".
   - Kliknij "Create pull request".

## Ocena

- Wykonanie wszystkich kroków: 0.5 punktu
- Poprawność i jakość kodu: 0.5 punktu
- Użycie Gita jest konieczne

Pamiętaj: Pytajcie o pomoc jeśli utkniecie, ale starajcie się rozwiązywać problemy samodzielnie!

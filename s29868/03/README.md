# Task 1: Podsumowanie eksperymentu z imputacją danych

## Symulacja braków
- Wprowadzono losowo **5%** i **20%** braków danych.

## Wyniki
- **5% braków:** wszystkie metody działają dobrze, różnice w AUC są niewielkie.  
- **20% braków:**  
  - **Mean:** prosty, mniej dokładny przy większych brakach.  
  - **KNN:** lepsza dokładność, szczególnie dla klasy rzadkiej.  
  - **MICE:** najstabilniejsza metoda, dobrze zachowuje strukturę danych.

## Wnioski
- Imputacja wpływa na jakość modelu, szczególnie przy większych brakach.  
- Przy małych brakach każda metoda jest wystarczająca.  
- Przy większych brakach rekomendowane są metody zaawansowane (KNN, MICE).  


# Task 2: Podsumowanie wyników modeli regresji logistycznej

- **Model bazowy (Base):**  
  Wysoka precyzja dla klasy dominującej, ale bardzo niski recall dla klasy rzadkiej. Model prawie nie wykrywa rzadkich przypadków przy standardowym progu 0.5.

- **Under / SMOTE / Balanced:**  
  Wszystkie trzy metody poprawiają wykrywalność klasy mniejszościowej (wyższy recall), kosztem spadku ogólnej dokładności i precyzji.

- **Wpływ progu klasyfikacji:**  
  - **Niski próg (0.2):** modele częściej przewidują klasę rzadką → recall rośnie, precyzja spada.  
  - **Wysoki próg (0.8):** modele stają się bardziej konserwatywne → precyzja rośnie, recall spada.
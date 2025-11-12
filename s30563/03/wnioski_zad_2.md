# Niezbalansowane klasy – wnioski

Model bazowy pozornie wydaje się najlepszy, gdyż utrzymuje 
bardzo wysoką precyzję, czułość oraz F1. Jednak gdy spojrzymy 
na wyniki dla klasy, która występuje rzadziej, widać 
wyraźną przewagę innych modeli. Modele z ważeniem klas, 
SMOTE oraz RUS radzą sobie znacznie lepiej w przypadku 
klasy mniejszościowej, co potwierdza, że model bazowy jest 
tylko pozornie dobry i traci zdolność do prawidłowego 
przewidywania dla takiego podziału danych.
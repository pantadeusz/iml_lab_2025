## Podsumowanie wyników (Lab 5)

- Sklearn (LogReg, baseline):
  - Walidacja: acc ≈ 0.991; bardzo mało FP/FN (cm: [[42,0],[1,71]]).
  - Test: acc ≈ 0.991; praktycznie perfekcyjnie (cm: [[42,1],[0,71]]).
- DNN (baseline):
  - Walidacja: acc ≈ 0.982 (cm: [[42,0],[2,70]]).
  - Test: acc ≈ 0.965 (cm: [[41,2],[2,69]]).
- DNN (po tunerze):
  - Walidacja: acc ≈ 0.974 (cm: [[42,0],[3,69]]).
  - Test: acc ≈ 0.974 (cm: [[42,1],[2,69]]).

## Wnioski
- Tuner wyraźnie poprawił DNN względem bazowego (0.965 → 0.974 acc na teście).
- Mimo poprawy DNN, model sklearn nadal wygrywa na teście (0.991 vs 0.974).
- Krótki budżet (mało prób/epok) wystarczył, by podbić wyniki DNN, ale do przebicia baseline’u potrzeba więcej strojenia (np. więcej prób, dłuższy trening, inne zakresy hp).
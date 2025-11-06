- Dane: `load_breast_cancer`, podział 80/20 (stratyfikowany).
- Modele:
  - RandomForest (200 drzew, bez skalowania).
  - Prosty MLP (64→32→2, ReLU), skalowanie StandarScalerem, 20 epok full‑batch.

## Wyniki (test)
- RF: acc=0.956, ROC AUC=0.993, AP=0.996
- DNN: acc=0.947, ROC AUC=0.988, AP=0.993

## Wnioski
- RandomForest minimalnie wygrywa z MLP na danych tabelarycznych (wyższa accuracy i AUC/AP).
- DNN również daje bardzo dobre wyniki, ale wymaga skalowania i treningu; RF działa „z pudełka”.
- Na takich cechach tabelarycznych RF to solidny baseline i trudny do pobicia przez prosty MLP.
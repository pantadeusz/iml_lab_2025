# Wyniki

Wyniki modelu znacznie polepszyli się po dodaniu warstwy konwulucyjnej i przy analizie problemów zamiany warstwy aktywacji ReLu na Elu

### Problemy z którymi spotkałem się podczas robienia zadania:

- Relu w Encoderze w Dense -> Zabija wszystkie ujemne wyniki i widzimy czarny kwadrat, a nie obrazek -> Usunęłem warstwę aktywacji w Densę
- Relu zabija progres uczenia się -> We wszystkich warstwach aktywacji zamieniłem wszystkie ReLu na Elu
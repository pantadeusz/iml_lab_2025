Na początku programu, generujemy dane które dzielimy na 95% klasy 0 i 5% klasy 1, dzięki weights=[0.95,0.05]

Następnie poddajemy te dane działaniom 3 modeli.

Model z ważeniem klas (class_weight='balanced') zwiększa wagę mniejszej klasie, przez co poprawia to recall w klasie 1

Model SMOTE, przetwarza zbiór treningowy, gdzie klasy są zrównoważone. Robi tak poprzez analizowanie istniejących próbek mniejszości i generuje nowe próbki, podobne do tej klasy by zrównoważyć ilość do próbek większości.

Model z undersamplingiem (RandomUnderSampler) usuwa losowo wybrane próbki z klasy większości, próbując je zrównać ilościowo z klasą mniejszośći.

Wszystkie te modele dążą do poprawy wartości recall.
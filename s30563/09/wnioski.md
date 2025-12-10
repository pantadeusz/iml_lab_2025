# Wnioski
Autoencoder ogólnie dobrze radzi sobie z przekształceniami 
obrazu zarówno w wersji konwolucyjnej, jak i podstawowej (tylko dense).
w tym zbiorze danych

Niemniej pojawiły się różnice. Model podstawowo y gorzej radził sobie z ogólnymi
kształtami, jednak zachowywał dużo szczegółów, takich jak wzory na koszulce.
Model konwolucyjny z kolei o wiele lepiej radził sobie z kształtami
ubrań, jednak praktycznie ignorował szczegóły na nich widoczne, co mogło
być spowodowane zmniejszeniem już i tak małego obrazu podczas encodingu 
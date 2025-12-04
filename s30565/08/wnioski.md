====================================
PODSUMOWANIE WYNIKÓW (accuracy)
====================================
Baseline / test oryginalny        : 0.9669
Baseline / test z augmentacją     : 0.4717
Aug model / test oryginalny       : 0.9460
Aug model / test z augmentacją    : 0.8284
Conv model / test oryginalny      : 0.9749
Conv model / test z augmentacją   : 0.9481
====================================

wnioski, ewidentnie gorsze wyniki na testowym zbiorze z augmentacja,
widac także ze gdy zrobimy augmentacje danych treningowych to równiez accuracy spada,
dla testu z augmentacją i bez. chociaż w momencie jak nauczymy model na danych augmentowanych
to accuracy na danych treningowych augmentowanych wyraźnie się zwiększyła wzgledem baseline.
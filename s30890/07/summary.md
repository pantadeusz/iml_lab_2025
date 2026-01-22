Prosta trójswarstwowa sieć (bez normalizacji) 
- Bez normalizacji: **94.44%** dokładności  
- Z warstwą normalizującą: **97.22%** dokładności  

8 małych konfiguracji sieci, różne rozmiary warstw i regularyzację.  
Modele, które osiągnęły **100% dokładności**, to:

- `C_8`
- `F_8_dropout`
- `G_4`
- `H_4_l2_dropout`

Najmniejszy model miał rozmiar ok. **24–25 KB**, więc nawet bardzo mała architektura jest w stanie osiągnąć pełną dokładność.

**Wnioski:**  
Warstwa normalizująca poprawia jakość uczenia.  
Zbiór jest prosty, dlatego nawet niewielkie sieci mogą osiągać 100%.  
Regularyzacja nie zawsze pomaga — czasem obniża wyniki.  

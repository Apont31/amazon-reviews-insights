# C5.1.1 — Analyse du besoin

## Problématique
Aller au-delà du sentiment binaire pour extraire **émotions**, **aspects produit** et **ironie** à partir des avis Amazon (*Electronics*).

## Contexte
- Données : C:\Users\antoi\OneDrive\Documents\Ynov\Projet fil rouge\Bloc 5\amazon-reviews-insights\data\amazon_insights.csv
- Schéma normalisé : `review_body`, `star_rating`, `review_date`, `review_title`, `product_title`, `product_id`
- Taille : 1,314,720 avis

## Contraintes
- Exécution locale, latence démo < 200 ms (baseline)
- Reproductibilité (notebooks, configs), anonymisation

## Points de vigilance (issus de l'EDA)
- Textes courts (<20 chars) : 10.34%
- Notes min/max : 1.0..5.0
- Dates présentes : True; Verified : True; Helpful : True

## KPI cibles
- Émotions F1-macro ≥ 0.60 ; ABSA F1-macro ≥ 0.60 ; Sarcasme AUC ≥ 0.75 ; Latence ≤ 200 ms ; Couverture ≥ 95%

## Décisions supportées
- Priorisation d'améliorations produit, enrichissement des fiches, priorités SAV, veille des tendances.

(Annexes : figures dans `docs/figs/`)

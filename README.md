# FibroPred: Revolucionant la Predicció Pronòstica en la Fibrosi Pulmonar

![Imatge del projecte](./portada.jpg)

La fibrosi pulmonar és una malaltia greu que provoca una cicatrització irreversible del pulmó, fent que una predicció precisa i primerenca sigui essencial. Els models actuals, com el GAP score, tenen un poder predictiu limitat, especialment en casos no idiopàtics. Aquí és on entra en joc *FibroPred*.  

FibroPred és una eina predictiva impulsada per intel·ligència artificial dissenyada per oferir pronòstics personalitzats des del moment del diagnòstic. Utilitzant algoritmes avançats com XGBoost i incorporant biomarcadors innovadors com la longitud telomèrica i l'agregació familiar, FibroPred supera els predictors tradicionals com l'edat i la funció pulmonar.  

La nostra solució es basa en conjunts de dades minuciosament preprocessats, que reflecteixen el trajecte real dels pacients en tres etapes: en el moment del diagnòstic, després d'un any i després de dos anys de tractament. Aquest enfocament progressiu garanteix prediccions realistes i adaptades al camí individual de cada pacient.  

Per generar confiança clínica, FibroPred inclou explicacions basades en SHAP, mostrant com cada variable influeix en el resultat, tant a nivell global com en cada cas particular.  

La nostra interfície web intuïtiva ofereix als professionals de la salut:  
- Estadístiques completes i visualitzacions de dades dels pacients  
- Entrada de dades en temps real per a nous pacients  
- Prediccions personalitzades amb gràfics interpretatius  
- Reentrenament del model sota demanda per a una millora contínua  

Amb FibroPred, els clínics disposen d'una eina potent de suport a la presa de decisions, permetent estratègies de tractament més primerenques, precises i personalitzades, millorant així els resultats dels pacients amb fibrosi pulmonar.  

## Sobre nosaltres:
Aquest projecte ha estat realitzat en el marc de la Hackaton BitsXLaMarató 2024 per les malalties respiratòries.

Els integrants de l'equip són: Cristina Aguilera, Pablo Vega, Iker Garcia, Sígrid Vila.


## Estructura del model:
Degut a la quantitat i sensibilitat de les dades disponibles, hem optat per models senzills i completament explicables per predir mètriques com: mort, progressió de la malaltia i necessitat de transplantament. El model usat ha set un XGBoost amb els hiperparàmetres optimitzats. Tot i això, la part important és l'explicabilitat del model, tant de manera general com indivdualitzada. Per això, hem realitat models tenint en compte els següents factors:

- Quantitat visites realitzades al metge (hi ha més mètriques com més visites realitzades).
- Mètrica a predir.

## Preprocessat de dades:
Altra vegada, degut a la sensibilitat de les dades ha calgut fer un preprocessat rigurós per tal de preservar la integritat i la explicabilitat del model. Més informació a la web creada amb stramlit.


## Estructura dels fitxers:

        .
        ├── content                         # Dades.
        ├── images_general                  # Images explicabilitat general.
        ├── model_results                   # CSVs de les dades dels models.
        ├── models                          # Models pytorch.
        ├── models_pesos                    # Pesos des models.
        ├── pages                           # Codi streamlit.
        ├── tmp                             # Imatges explicabilitat indiviual.
        ├── utils                           # Codi entrenament models.
        ├── weights                         # Pesos model pytorch.  
        ├── FibroPred.py                    # Pàgina general stramlit.
        ├── config.py                       # Variables globals.
        ├── data_preprocessing.py           # Preprocessat de dades.
        ├── requirements.py                 # Requeriments.
        └── train_autoencoder.py            # Entrnament autoencoder model pytorch.
      

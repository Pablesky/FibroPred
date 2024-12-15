import streamlit as st

st.title("Processament de Dades per a FibroPred")

# Mostrar los datos actuales
st.subheader("Particularitats de les Dades Mèdiques")


st.markdown("Les dades mèdiques són altament sensibles i complexes a causa del seu impacte directe en l'atenció i els resultats dels pacients. Contenen registres confidencials que han de ser protegits amb molta cura segons les regulacions de privacitat estrictes. A més de les preocupacions de privacitat, la precisió d'aquestes dades és crucial, ja que fins i tot errors menors poden provocar diagnòstics incorrectes i tractaments potencialment mortals. Els conjunts de dades mèdiques sovint inclouen entrades incompletes, inconsistents o errònies a causa dels processos d’entrada manual, cosa que requereix una validació, correcció i neteja exhaustives. A més, la diversitat de tipus de dades —des de mesures clíniques i resultats d'imatges fins a informació genètica— afegeix una capa addicional de complexitat, requerint experiència específica per a una interpretació i processament adequats.")


st.subheader("Comprensió de les Dades")

st.markdown("Abans d'iniciar la fase de processament, vam col·laborar estretament amb experts mèdics per entendre els diferents tipus de variables presents en el conjunt de dades. Aquestes variables inclouen:")


st.markdown(" - **Dades Demogràfiques:** Edat, gènere i antecedents familiars.")

st.markdown(" - **Mesures Clíniques:** Mètriques de la funció pulmonar com DLCO (%), FVC (%) i FVC (L).")

st.markdown(" - **Dades Radiològiques:** Patrons identificats en imatges com el UIP.")

st.markdown(" - **Informació del Tractament:** Ús de fàrmacs antifibròtics i antiinflamatoris.")

st.markdown(" - **Indicadors de Progressió de la Malaltia:** Canvis en les mètriques pulmonars i supervivència dels pacients.")



st.subheader("Passos de Processament de Dades")


st.write("#### Generació de Noves Columnes")
st.markdown("L'edat en el moment del diagnòstic es va categoritzar en grups per identificar factors de risc associats a diferents franges d'edat. Vam combinar 'UIP' i 'Probable UIP' en una sola categoria per simplificar les prediccions. Es va crear una columna binària per indicar si un pacient havia rebut tractaments antiinflamatoris específics. Es van afegir noves columnes per fer un seguiment dels canvis en DLCO (%) i FVC (%) entre el diagnòstic i les etapes de seguiment. Els canvis significatius es van marcar com a indicadors binaris. Basant-nos en les dates rellevants, vam determinar l'estat de trasplantament i de defunció per millorar l'anàlisi de supervivència. Les entrades de mutacions genètiques es van verificar i es van convertir en un format binari.")


st.write("#### Actualització de Columnes Existents")
st.markdown("Basant-nos en els resultats diagnòstics binaris, es va actualitzar el diagnòstic final. Es va marcar la implicació extrapulmonar segons les anormalitats telomèriques. La presència de malalties hematològiques i hepàtiques es va actualitzar en funció de les entrades clíniques.")


st.write("#### Normalització de Dades")
st.markdown("Les columnes amb respostes 'Sí/No' es van estandarditzar a valors binaris (1/0).")


st.write("#### Eliminació de Columnes Innecessàries")
st.markdown("Per reduir la complexitat de les dades i millorar el rendiment del model, es van eliminar columnes irrellevants o redundants, inclosos detalls dels pacients, tractaments específics i diverses notes.")


st.subheader("Creació de Tres Conjunts de Dades Diferenciats")
st.markdown(" - En el moment del diagnòstic, el conjunt de dades inclou només la informació bàsica dels pacients disponible en el moment del diagnòstic. La predicció en les primeres etapes és crucial per identificar pacients d’alt risc i iniciar tractaments adequats com més aviat millor. Això permet als clínics predir la progressió de la malaltia fins i tot abans que apareguin símptomes significatius, facilitant una intervenció primerenca.")
st.markdown(" - Després d'un any, el conjunt de dades s'amplia amb dades clíniques addicionals recopilades un any després del diagnòstic. Aquesta etapa proporciona una visió a mitjà termini de la progressió de la malaltia, ajudant els clínics a ajustar els plans de tractament segons la resposta inicial del pacient.Després de dos anys, el conjunt de dades més complet inclou totes les dades de seguiment disponibles després de dos anys. Aquest conjunt de dades admet la modelització del pronòstic a llarg termini, proporcionant una comprensió més completa de la progressió de la malaltia i dels resultats dels pacients.")
st.markdown(" - Després de dos anys, el conjunt de dades més complet inclou totes les dades de seguiment disponibles després de dos anys. Aquest conjunt de dades admet la modelització del pronòstic a llarg termini, proporcionant una comprensió més completa de la progressió de la malaltia i dels resultats dels pacients.")
st.markdown(" - Entrenant tres models separats amb aquests conjunts de dades diferenciats, ens vam assegurar que FibroPred pogués oferir prediccions precises i específiques per a cada etapa. Aquest enfocament millora l’aplicabilitat clínica de l’eina, permetent ajustaments personalitzats del tractament durant tot el procés de gestió de la malaltia.")

st.image('media/foto_processament.jpg')















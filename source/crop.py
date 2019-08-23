Oggetto: Ricevimento progetto TTR sui videogiochi

Messaggio:
Buongiorno professore,

Sono Marco Emporio e sto lavorando al progetto di ttr con Adriano Tumminelli. 
Abbiamo raggiunto buoni risultati nel progetto di riconoscimento videogiochi da screenshot
e volevamo chiederle se si potesse fissare un ricevimento a inizio settembre per avere
qualche consiglio su come migliorare/espandere il progetto e se siamo sulla strada giusta.
Le elenchiamo ciò che abbiamo fatto finora:

-) Raccolto 100+ screenshot da più di 10 videogiochi (alcuni abbastanza simili tra loro,
 appartenenti allo stesso genere) e diviso training set e testing set

-) Estratto le features utilizzando varie reti con keras/tensorflow
    vgg16, vgg19, inception, mobilenet, resnetv2, NAS, resnet, resnext, etc.

-) Addestrato un modello con una SVM

-) Classificato gli screenshot del testing set e confrontati col ground truth, 
creando una matrice di confusione e un report dettagliato dei casi in cui fallisce la classificazione.

-) Rieseguiti i punti precedenti ma con le immagini ritagliate nei 4 angoli e creati i report anche di questi

Dai dati raccolti abbiamo notato che le reti più performanti con immagini complete (mobilenet) che 
classifica al 98% in modo corretto, non ha miglioramenti col crop. 
Mentre le reti con punteggio peggiore su immagini complete (vgg16), ottengono molti più benefici dal crop
si passa dal 33% al 65%.

Nel frattempo provvediamo a espandere il dataset aggiungendo altri giochi simili a quelli che abbiamo
in modo da abbassare il punteggio di mobilenet con screenshot completi


Adriano e Marco
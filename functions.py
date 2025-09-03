#Here are the functions for preprocessing the binary model for edibility
###############IMPORTS#######################
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import resample
import tensorflow as tf
from keras import layers, models
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping

#creating a function to add edibility criteria and drop duplicates
def add_edibility(df) :
    df=df.drop_duplicates()
    species_to_label = {
    "Agaricus augustus": 0,
    "Agaricus xanthodermus": 1,
    "Amanita amerirubescens": 1,
    "Amanita augusta": 1,
    "Amanita brunnescens": 1,
    "Amanita calyptroderma": 0,
    "Amanita citrina": 1,
    "Amanita flavoconia": 1,
    "Amanita muscaria": 1,
    "Amanita pantherina": 1,
    "Amanita persicina": 1,
    "Amanita phalloides": 1,
    "Amanita rubescens": 1,
    "Amanita velosa": 1,
    "Apioperdon pyriforme": 0,
    "Armillaria borealis": 0,
    "Armillaria mellea": 0,
    "Armillaria tabescens": 0,
    "Artomyces pyxidatus": 0,
    "Bjerkandera adusta": 1,
    "Bolbitius titubans": 1,
    "Boletus edulis": 0,
    "Boletus pallidus": 0,
    "Boletus reticulatus": 0,
    "Boletus rex-veris": 0,
    "Calocera viscosa": 1,
    "Calycina citrina": 1,
    "Cantharellus californicus": 0,
    "Cantharellus cibarius": 0,
    "Cantharellus cinnabarinus": 0,
    "Cerioporus squamosus": 0,
    "Cetraria islandica": 0,
    "Chlorociboria aeruginascens": 1,
    "Chlorophyllum brunneum": 1,
    "Chlorophyllum molybdites": 1,
    "Chondrostereum purpureum": 1,
    "Cladonia fimbriata": 1,
    "Cladonia rangiferina": 0,
    "Cladonia stellaris": 0,
    "Clitocybe nebularis": 1,
    "Clitocybe nuda": 0,
    "Coltricia perennis": 1,
    "Coprinellus disseminatus": 1,
    "Coprinellus micaceus": 0,
    "Coprinopsis atramentaria": 1,
    "Coprinopsis lagopus": 1,
    "Coprinus comatus": 0,
    "Crucibulum laeve": 1,
    "Cryptoporus volvatus": 1,
    "Daedaleopsis confragosa": 1,
    "Daedaleopsis tricolor": 1,
    "Entoloma abortivum": 0,
    "Evernia mesomorpha": 1,
    "Evernia prunastri": 1,
    "Flammulina velutipes": 0,
    "Fomes fomentarius": 1,
    "Fomitopsis betulina": 1,
    "Fomitopsis mounceae": 1,
    "Fomitopsis pinicola": 1,
    "Galerina marginata": 1,
    "Ganoderma applanatum": 1,
    "Ganoderma curtisii": 1,
    "Ganoderma oregonense": 1,
    "Ganoderma tsugae": 0,
    "Gliophorus psittacinus": 1,
    "Gloeophyllum sepiarium": 1,
    "Graphis scripta": 1,
    "Grifola frondosa": 0,
    "Gymnopilus luteofolius": 1,
    "Gyromitra esculenta": 1,
    "Gyromitra gigas": 1,
    "Gyromitra infula": 1,
    "Hericium coralloides": 0,
    "Hericium erinaceus": 0,
    "Hygrophoropsis aurantiaca": 1,
    "Hypholoma fasciculare": 1,
    "Hypholoma lateritium": 0,
    "Hypogymnia physodes": 1,
    "Hypomyces lactifluorum": 0,
    "Imleria badia": 0,
    "Inonotus obliquus": 0,
    "Ischnoderma resinosum": 1,
    "Kuehneromyces mutabilis": 0,
    "Laccaria ochropurpurea": 1,
    "Lactarius deliciosus": 0,
    "Lactarius torminosus": 1,
    "Lactarius turpis": 1,
    "Laetiporus sulphureus": 0,
    "Leccinum albostipitatum": 0,
    "Leccinum aurantiacum": 0,
    "Leccinum scabrum": 0,
    "Leccinum versipelle": 0,
    "Lepista nuda": 0,
    "Leratiomyces ceres": 1,
    "Leucoagaricus americanus": 1,
    "Leucoagaricus leucothites": 0,
    "Lobaria pulmonaria": 1,
    "Lycogala epidendrum": 1,
    "Lycoperdon perlatum": 0,
    "Lycoperdon pyriforme": 0,
    "Macrolepiota procera": 0,
    "Merulius tremellosus": 1,
    "Mutinus ravenelii": 1,
    "Mycena haematopus": 1,
    "Mycena leaiana": 1,
    "Nectria cinnabarina": 1,
    "Omphalotus illudens": 1,
    "Omphalotus olivascens": 1,
    "Panaeolus papilionaceus": 1,
    "Panellus stipticus": 1,
    "Parmelia sulcata": 1,
    "Paxillus involutus": 1,
    "Peltigera aphthosa": 1,
    "Peltigera praetextata": 1,
    "Phaeolus schweinitzii": 1,
    "Phaeophyscia orbicularis": 1,
    "Phallus impudicus": 0,
    "Phellinus igniarius": 1,
    "Phellinus tremulae": 1,
    "Phlebia radiata": 1,
    "Phlebia tremellosa": 1,
    "Pholiota aurivella": 1,
    "Pholiota squarrosa": 1,
    "Phyllotopsis nidulans": 1,
    "Physcia adscendens": 1,
    "Platismatia glauca": 1,
    "Pleurotus ostreatus": 0,
    "Pleurotus pulmonarius": 0,
    "Psathyrella candolleana": 1,
    "Pseudevernia furfuracea": 1,
    "Pseudohydnum gelatinosum": 0,
    "Psilocybe azurescens": 1,
    "Psilocybe caerulescens": 1,
    "Psilocybe cubensis": 1,
    "Psilocybe cyanescens": 1,
    "Psilocybe ovoideocystidiata": 1,
    "Psilocybe pelliculosa": 1,
    "Retiboletus ornatipes": 0,
    "Rhytisma acerinum": 1,
    "Sarcomyxa serotina": 0,
    "Sarcoscypha austriaca": 1,
    "Sarcosoma globosum": 1,
    "Schizophyllum commune": 1,
    "Stereum hirsutum": 1,
    "Stereum ostrea": 1,
    "Stropharia aeruginosa": 1,
    "Stropharia ambigua": 1,
    "Suillus americanus": 0,
    "Suillus granulatus": 0,
    "Suillus grevillei": 0,
    "Suillus luteus": 0,
    "Suillus spraguei": 0,
    "Tapinella atrotomentosa": 1,
    "Trametes betulina": 1,
    "Trametes gibbosa": 1,
    "Trametes hirsuta": 1,
    "Trametes ochracea": 1,s
    "Trametes versicolor": 1,
    "Tremella mesenterica": 0,
    "Trichaptum biforme": 1,
    "Tricholoma murrillianum": 0,
    "Tricholomopsis rutilans": 0,
    "Tylopilus felleus": 1,
    "Tylopilus rubrobrunneus": 1,
    "Urnula craterium": 1,
    "Verpa bohemica": 1,
    "Volvopluteus gloiocephalus": 0,
    "Vulpicida pinastri": 1,
    "Xanthoria parietina": 1,
    }
    df["edibility"] = df["label"].map(species_to_label)
    return df

#function to create a generator to use the image_path

def generator(df,x_col,y_col,img_size,batch_size,shuffle) :

    datagen = ImageDataGenerator(rescale=1./255)

    gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col=x_col,
    y_col=y_col,
    target_size=img_size,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=shuffle,
    )
    return gen

#function for compiling

def model_compiling(model) :
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','precision','recall'])


#function for fitting
def model_fitting(model,train_data,validation_data,epochs=10) :

    #callbacks
    es = [EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy")]

    #fitting
    history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    verbose=1,
    callbacks=es
    )

    return history

#function for evaluating
def evaluate_model(model,test_data) :
    evaluation=model.evaluate(test_data)
    return evaluation

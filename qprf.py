#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
from random import randrange
import sklearn
from sklearn import cross_validation
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
import cStringIO
from numpy import random
import scipy
from scipy.stats import chisquare
from copy import deepcopy
import operator 
import matplotlib
import io
from io import BytesIO
#matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image ## Hide for production
from collections import OrderedDict

app = Flask(__name__, static_url_path = "")

"""
    JSON Parser for Read Across
"""
def getJsonContentsQPRF (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]
        #print "\n\n\n ok \n\n\n"
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
        metaInfo = dataset.get("meta", None)
        features = dataset.get("features", None)
        #print "\n\n\n ok \n\n\n"
        qprf_creator = metaInfo["creators"][0]

        substanceURI = parameters.get("substanceURI", None) # ID for QPRF
        predictedFeature = parameters.get("predictedFeature", None)
        structures = parameters.get("structures", None)
        algorithm = parameters.get("algorithm", None)
        #print "\n\n\n ok \n\n\n"
        doaURI = parameters.get("doaURI", None)
        doaMethod = parameters.get("doaMethod", None)
        print doaURI, "\n\n\n"
        for i in range (len(dataEntry)):
            if dataEntry[i]["compound"]["URI"] == substanceURI and predictedFeature in dataEntry[i]["values"]:
                doaValue = dataEntry[i]["values"][doaURI]
        doaALL = [doaValue, doaMethod]
        #print "\n\n\n ok \n\n\n"
        substance = {}
        substance["uri"] = substanceURI
        substance["inchi"] = structures[0]["Std. InChI"]
        substance["ec"] = structures[0]["EC number"]
        substance["cas"] = structures[0]["CasRN"]
        substance["iupac"] = structures[0]["IUPAC name"]
        substance["reach"] = structures[0]["REACH registration date"]
        #print "\n\n\n ok5 \n\n\n"
        prediction = {}
        for i in range (len(features)):
            if features[i]["uri"] == predictionFeature:
                prediction["name"] = features[i]["name"]
                break
        prediction["uri"] = predictionFeature
        prediction["model"] = algorithm["meta"]["titles"][0]
        #print "\n\n\n ok6 \n\n\n"
        for i in range (len(dataEntry)):
            if dataEntry[i]["compound"]["URI"] == substanceURI:
                if predictedFeature in dataEntry[i]["values"].keys(): 
                    prediction["value"] = dataEntry[i]["values"][predictedFeature]
                    break

        #prediction["descriptors"] 
                
        #print "\n\n\n ok7 \n\n\n"
        variables = dataEntry[0]["values"].keys() 
        variables.sort()  # NP features including predictionFeature

        datapoints =[] # list of nanoparticle feature vectors not for qprf
        nanoparticles=[] # nanoparticles not substanceURI 

        for i in range(len(dataEntry)-2): ##?
            datapoints.append([])

        qprf_datapoints = []

        #print "\n\n\n ok8 \n\n\n"
        counter = 0
        for i in range(len(dataEntry)):

            if dataEntry[i]["compound"].get("URI") != substanceURI:
                nanoparticles.append(dataEntry[i]["compound"].get("URI"))
                for j in variables:
                    if j != predictionFeature:
                        datapoints[counter].append(dataEntry[i]["values"].get(j))
                counter+=1
            elif predictedFeature in dataEntry[i]["values"]:
                for j in variables:
                    if j != predictedFeature and j!= predictionFeature:
                        #print dataEntry[i]["values"].get(j), j
                        qprf_datapoints.append(dataEntry[i]["values"].get(j))
        #print variables, "\n\n\n\n"
        variables.remove(predictionFeature) 

        variable_names = []
        for i in range (len(variables)):
            for j in range (len(features)):
                if features[j]["uri"] == variables[i]:
                    variable_names.append( features[j]["name"])

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
    #print len(nanoparticles), len(read_across_datapoints)
    #print readAcrossURIs, read_across_datapoints
    #return variables, datapoints, read_across_datapoints, predictionFeature, target_variable_values, byteify(readAcrossURIs), nanoparticles
    return substance, prediction, nanoparticles, datapoints, qprf_datapoints, variables, variable_names, doaALL, qprf_creator


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


"""
    [[],[]]  Matrix to dictionary for Nearest Neighboura
"""
"""
def mat2dicNN(matrix, name):
    myDict = {}
    for i in range (len (matrix[0])):
        myDict[name + " NN_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
    return byteify(myDict)
"""

"""
    [[],[]]  Matrix to dictionary 
"""
"""
def mat2dic(matrix):
    myDict = {}
    for i in range (len (matrix)):
        myDict["Row_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
    return byteify(myDict)
"""

"""
    [[]]  Matrix to dictionary Single Row
"""
"""
def mat2dicSingle(matrix):
    myDict = {}
    myDict["Row_1"] = matrix
    return byteify(myDict)
"""

"""
    Normaliser
"""
def manual_norm(myTable, myMax, myMin):
    #print myTable, "\n\n\n", myMax, "\n\n\n", myMin
    if myMax>myMin:
        for i in range (len(myTable)):
            myTable[i] = (myTable[i]-myMin)/(myMax-myMin)
    else:
        for i in range (len(myTable)):
            myTable[i] = 0
    return myTable

"""
    Distances
"""
def distances (read_across_datapoints, datapoints, variables, readAcrossURIs, nanoparticles):

    datapoints_transposed = map(list, zip(*datapoints)) 

    RA_datapoints_transposed = map(list, zip(*read_across_datapoints)) ###
    #RA_datapoints_transposed = read_across_datapoints###

    for i in range (len(datapoints_transposed)):
        max4norm = numpy.max(datapoints_transposed[i])
        min4norm = numpy.min(datapoints_transposed[i])

        datapoints_transposed[i] = manual_norm(datapoints_transposed[i], max4norm, min4norm)

        RA_datapoints_transposed[i] = manual_norm(RA_datapoints_transposed[i], max4norm, min4norm) ###
    #RA_datapoints_transposed = manual_norm(RA_datapoints_transposed, max4norm, min4norm) ###


    term1 = []
    term2 = []
    for i in range (len(variables)):
        term1.append(0)
        term2.append(1)

    datapoints_norm = map(list, zip(*datapoints_transposed)) 

    RA_datapoints_norm = map(list, zip(*RA_datapoints_transposed))  ###
    #RA_datapoints_norm = RA_datapoints_transposed ###

    max_eucl_dist = euclidean_distances(term1, term2)
    #print RA_datapoints_norm
    eucl_dist = euclidean_distances(RA_datapoints_norm, datapoints_norm)
    eucl_dist = numpy.array(eucl_dist)
    eucl_dist = eucl_dist/max_eucl_dist
    eucl_dist = numpy.round(eucl_dist,4)


    np_plus_eucl = []
    for i in range (len(readAcrossURIs)):
        np_plus_eucl.append([nanoparticles, eucl_dist[i]]) 


    eucl_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_eucl[i][0], np_plus_eucl[i][1]
        np = zip (np_plus_eucl[i][1], np_plus_eucl[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        eucl_sorted.append([np_sorted, dist_sorted])
    #print "\n\nSorted\n\n", eucl_sorted
    ## [ [ [names] [scores] ] [ [N] [S] ]]
    ##       00      01          10  11    


    #eucl_transposed = map(list, zip(*eucl_sorted)) 
    eucl_dict = {} # []
    for i in range (len(readAcrossURIs)):
        #eucl_dict.append(mat2dicNN(eucl_sorted[i], readAcrossURIs[i])) #
        for j in range (len (eucl_sorted[i][0])):
            eucl_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [eucl_sorted[i][0][j], eucl_sorted[i][1][j]]
    eucl_dict = byteify(eucl_dict)

    max_manh_dist = metrics.pairwise.manhattan_distances(term1, term2)
    manh_dist = metrics.pairwise.manhattan_distances(RA_datapoints_norm, datapoints_norm)
    manh_dist = numpy.array(manh_dist)
    manh_dist = manh_dist/max_manh_dist
    manh_dist = numpy.round(manh_dist,4)

    np_plus_manh = []
    for i in range (len(readAcrossURIs)):
        np_plus_manh.append([nanoparticles, manh_dist[i]]) 

    manh_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_manh[i][0], np_plus_manh[i][1]
        np = zip (np_plus_manh[i][1], np_plus_manh[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        manh_sorted.append([np_sorted, dist_sorted])

    manh_dict = {}
    for i in range (len(readAcrossURIs)):
        #manh_dict.append(mat2dicNN(manh_sorted[i], readAcrossURIs[i]))
        for j in range (len (manh_sorted[i][0])):
            manh_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [manh_sorted[i][0][j], manh_sorted[i][1][j]]
    manh_dict = byteify(manh_dict)

    ensemble_dist = (eucl_dist + manh_dist)/2
    #print "Eucl.: ", eucl_dist, "\n Manh.: ", manh_dist,"\n Ens.: ", ensemble_dist

    np_plus_ens = []
    for i in range (len(readAcrossURIs)):
        np_plus_ens.append([nanoparticles, ensemble_dist[i]]) 

    ens_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_ens[i][0], np_plus_ens[i][1]
        np = zip (np_plus_ens[i][1], np_plus_ens[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        ens_sorted.append([np_sorted, dist_sorted])

    ens_dict = {}
    for i in range (len(readAcrossURIs)):
        #ens_dict.append(mat2dicNN(ens_sorted[i], readAcrossURIs[i]))
        for j in range (len (ens_sorted[i][0])):
            ens_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [ens_sorted[i][0][j], ens_sorted[i][1][j]]
    ens_dict = byteify(ens_dict)

    ### PLOT PCA
    #print datapoints_norm, "\n\n\n", RA_datapoints_norm

    pcafig = plt.figure()

    if len(datapoints_norm[0]) >=3:
        ax = pcafig.add_subplot(111, projection='3d')
        pca = decomposition.PCA(n_components=3)
        pca.fit(datapoints_norm)
        dt = pca.transform(datapoints_norm)
        ax.scatter(dt[:,0], dt[:,1], dt[:,2], c='r',  label = 'Original Values')

        RA_dt = pca.transform(RA_datapoints_norm)
        ax.scatter(RA_dt[:,0], RA_dt[:,1], RA_dt[:,2], c='b', label = 'QPRF Query Values')

        ax.set_xlabel("1st Principal Component") 
        ax.set_ylabel("2nd Principal Component")
        ax.set_zlabel("3rd Principal Component")
        ax.set_title("3D Projection of Datapoints")
    elif len(datapoints_norm[0]) ==2: ###
        pca = decomposition.PCA(n_components=2)
        pca.fit(datapoints_norm)
        dt = pca.transform(datapoints_norm)
        plt.scatter(dt[:,0], dt[:,1], c='r',  label = 'Original Values') 
    
        RA_dt = pca.transform(RA_datapoints_norm)
        plt.scatter(RA_dt[:,0], RA_dt[:,1], c='b', label = 'QPRF Query Values') 
    
        plt.xlabel("1st Principal Component" ) 
        plt.ylabel("2nd Principal Component") 
        #plt.title("2D Projection of Datapoints") 


    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)


    #plt.tight_layout()
    #plt.show() #HIDE show on production

    figfile = BytesIO()
    pcafig.savefig(figfile, dpi=300, format='png', bbox_inches='tight') #bbox_inches='tight'
    figfile.seek(0) 
    pcafig_encoded = base64.b64encode(figfile.getvalue())    
    
    return ens_sorted, pcafig_encoded

"""
    Predict
"""
"""
def RA_predict(euclidean, manhattan, ensemble, name, predictionFeature, nano2value):
    #print euclidean[0] # names of np
    #print euclidean[1] # dist values
    #print nano2value

    eu_score = 0
    ma_score = 0
    en_score = 0

    eu_div = 0
    ma_div = 0
    en_div = 0

    for i in range (len(euclidean[0])):
        if euclidean[1][i] < 1:
            eu_score += (1 - euclidean[1][i])*(nano2value[euclidean[0][i]]) #just the name
            eu_div += 1 - euclidean[1][i]
        if manhattan[1][i] < 1:
            ma_score += (1 - manhattan[1][i])*(nano2value[euclidean[0][i]]) #just the name
            ma_div += 1 - manhattan[1][i]
        if ensemble[1][i] < 1:
            en_score += (1 - ensemble[1][i])*(nano2value[euclidean[0][i]]) #just the name
            en_div += 1 - ensemble[1][i]
    eu_score = eu_score/eu_div
    ma_score = ma_score/ma_div
    en_score = en_score/en_div
    #print eu_score
    return [name, round(eu_score,2)], [name, round(ma_score,2)], [name, round(en_score,2)]
"""

"""
    Pseudo AD
"""
"""
def RA_applicability(euclidean, manhattan, ensemble, name):
    eu_score = 0
    ma_score = 0
    en_score = 0
    for i in range (len(euclidean[1])): # list of vals
        if euclidean[1][i] < 0.4:
            eu_score +=1
        if manhattan[1][i] < 0.33:
            ma_score +=1
        if ensemble[1][i] < 0.36:
            en_score +=1
    eu_score = eu_score/len(euclidean[1])
    ma_score = ma_score/len(euclidean[1])
    en_score = en_score/len(euclidean[1])
    #RA_appl = [["Euclidean", eu_score], ["Manhattan", ma_score], ["Ensemble", en_score]]
    #return ["Euclidean", eu_score], ["Manhattan", ma_score], ["Ensemble", en_score]
    return [name, eu_score], [name, ma_score], [name, en_score]
"""
    

@app.route('/pws/qprf', methods = ['POST'])
def create_task_qprf():

    if not request.json:
        abort(400)

    # DEFAULT VALUES
    substance_dict = OrderedDict()
    substance_dict = {
    "General" :["Instructions", "This section is aimed at defining the substance for which the (Q)SAR prediction is made."],
    "1.1": ["CAS number", "Report the CAS number."],
    "1.2": ["EC number", "Report the EC number."],
    "1.3": ["Chemical name", "Report the chemical names (IUPAC and CAS names)."],
    "1.4": ["Structural formula", "Report the structural formula."],
    "1.5 General": ["Structure codes", "Report available structural information for the substance, including the structure code used \
    to run the model. If you used a SMILES or InChI code, report the code in the corresponding field below. \
    If you have used any another format (e.g. mol file), please include the corresponding structural representation \
    as supporting information."],
    "1.5 a.": ["SMILES", "Report the SMILES of the substance (indicate if this is the one used for the model prediction)."],
    "1.5 b.": ["InChI", "Report the InChI code of the substance (indicate if this is the one used for the model prediction)."],
    "1.5 c.": ["Other structural representation", "Indicate if another structural representation was used to generate the prediction. \
    Indicate whether this information is included as supporting information. Example: 'mol file used \
    and included in the supporting information'."],
    "1.5 d.": ["Stereochemical features", "Indicate whether the substance is a stereo-isomer and consequently may have properties \
    that depend on the orientation of its atoms in space. Identify the stereochemical features that may affect the reliability \
    of predictions for the substance, e.g. cis-trans isomerism, chiral centres. Are these features encoded in the structural \
    representations mentioned above?"]
    }

    general_dict = OrderedDict()
    general_dict = {
    "General" :["Instructions", "General information about the compilation of the current QPRF is provided in this section."],
    "2.1": ["Date of QPRF", "Report the date of compilation of the QPRF. Example: '01 January 2007'."],
    "2.2" : ["QPRF author and contact details", "Report the contact details of the author of the QPRF."]
    }
    prediction_dict = OrderedDict()
    prediction_dict = {
    "General" :["Instructions", "The information provided in this section will help to facilitate considerations on the \
    scientific validity of the model (as defined in the OECD Principles for the validation of (Q)SAR models) \
    and the reliability of the prediction. Detailed information on the model are stored in the corresponding \
    QMRF which is devised to reflect as much as possible the OECD principles. Remember that the QMRF and the \
    QPRF are complementary, and a QPRF should always be associated with a defined QMRF."],
    "3.1 General": ["Endpoint", "(OECD Principle 1)"], 
    "3.1 a.": ["Endpoint", "Define the endpoint for which the model provides predictions (this information should correspond\
    to the information provided in the QMRF under fields 3.2 and 3.3). Example: 'Nitrate radical degradation \
    rate constant KNO3'."],
    "3.1 b.": ["Dependent variable", "Report the dependent variable for which the model provides predictions including \
    any transformations introduced for modelling purposes (note that this information should correspond to the \
    information provided in the QMRF under field 3.5). Example: '-log (KNO3)'."],
    "3.2 General" : ["Algorithm",  "(OECD Principle 2)"],
    "3.2 a.": ["Model or submodel name", "Identify the model used to make the prediction and possibly report its name as stored \
    in the corresponding QMRF; in the QMRF the model name is reported in the field QSAR identifier. Examples: \
    'BIOWIN for Biodegradation'; 'TOPKAT Developmental Toxicity Potential'. If applicable identify the specific \
    submodel or algorithm applicable to the specific chemical Examples: 'BIOWIN 1'; 'TOPKAT Skin Irritation Acyclics \
    (Acids, Amines, Esters) MOD v SEV Model'; 'ECOSAR esters model'."],
    "3.2 b.": ["Model version", " Identify, where relevant, the version number and/or date of the model and submodel."],
    "3.2 c.": ["Reference to QMRF", " Provide relevant information about the QMRF that stores information about the model \
    used to make the prediction. Possible useful pieces of information are: availability, source, reference number (if any) \
    of the QMRF. Examples: 'The corresponding QMRF named -BIOWIN for Biodegradation- has been downloaded from the JRC QSAR \
    Model Database'; 'The corresponding QMRF named -TOPKAT Skin Irritation Acyclics (Acids, Amines, Esters) MOD v SEV Model-\
    has been newly compiled'."],
    "3.2 d.": ["Predicted value (model result)", " Report the predicted value (including units) obtained from the application \
    of the model to the query chemical. For an expert system such as Derek for Windows, report the alert triggered together\
    with the reasoning. Example: ' aromatic amine - mutagenicity, plausible'."],
    "3.2 e.": ["Predicted value (comments)", " If the result is qualitative (e.g. yes/no) or semi-quantitative (e.g. low/medium/high), \
    explain the cut-off values that were used as the basis for classification. In reporting the predicted value, pay attention \
    to the transformations (e.g. if the prediction is made in log units, apply anti-logarithm function)."],
    "3.2 f.": ["Input for prediction", " Specify what kind of input was used to generate the prediction (SMILES, mol file, graphical \
    interface etc). Please provide the structure code used to generate the prediction (unless already provided in section 1.5)."],
    "3.2 g.": ["Descriptor values", " Where appropriate, report the values (experimental or calculated data) for numerical descriptors \
    and indicate which values were used for making the prediction."],
    "3.3 General": ["Applicability domain", "(OECD principle 3)"],
    "3.3 a.": ["Domains", "Discuss whether the query chemical falls in the applicability domain of the model as defined in the corresponding \
    QMRF (section 5 of QMRF, Defining the applicability domain - OECD Principle 3). If additional software/methods were used to assess \
    the applicability domain then they should also be documented in this section. Include a discussion about: i. descriptor domain \
    ii. structural fragment domain (e.g., discuss whether the chemical contains fragments that are not represented in the model \
    training set) iii. mechanism domain (discuss whether the chemical is known or considered to act according to the mechanism of \
    action associated with the used model) iv. metabolic domain, if relevant"],
    "3.3 b.": ["Structural analogues", "List the structural analogues that are present in the training or test sets, or accessible from \
    other sources (in this case you should explain how the structural analogue was retrieved ) and why they are considered \
    analogues). For each analogue, report the CAS number, the structural formula, the SMILES code, and the source (e.g., \
    training set, test set or other source). For an expert system (like Derek for Windows or TOPKAT), the example compounds or \
    structurally related analogues with their experimental data should be provided here. "],
    "3.3 c." : ["Considerations on structural analogues", "Discuss how predicted and experimental data for analogues support the \
    prediction of the chemical under consideration. "],
    "3.4": ["The uncertainty of the prediction (OECD principle 4)", "If possible, comment on the uncertainty of the prediction for \
    this chemical, taking into account relevant information (e.g. variability of the experimental results). "],
    "3.5": ["The chemical and biological mechanisms according to the model underpinning the predicted result (OECD principle 5)",  
    "Discuss the mechanistic interpretation of the model prediction for this specific chemical. For an expert system based on \
    structural alerts (e.g. Derek for Windows, OncologicTM) the rationale for the structural alert fired should be provided."],
    }
    
    adequacy_dict = OrderedDict()
    adequacy_dict = {
    "General" :["Instructions", "The information provided in this section might be useful, depending on the reporting needs \
    and formats of the regulatory framework of interest. \
    This information aims to facilitate considerations about the adequacy of the (Q)SAR prediction (result) \
    estimate. A (Q)SAR prediction may or may not be considered adequate ('fit-for-purpose'), depending on \
    whether the prediction is sufficiently reliable and relevant in relation to the particular regulatory \
    purpose. The adequacy of the prediction also depends on the availability of other information, and is \
    determined in a weight-of-evidence assessment."],
    "4.1": ["Regulatory purpose", "Explain the regulatory purpose for which the prediction described \
    in Section 3 is being used."],
    "4.2": ["Approach for regulatory interpretation of the model result", "Describe how the predicted result \
    is going to be interpreted in light of the specific regulatory purpose (e.g. by applying an algorithm or \
    regulatory criteria). This may involve the need to convert the units of the dependent variable (e.g. from \
    log molar units to mg/l). It may also involve the application of another algorithm, an assessment factor, \
    or regulatory criteria, and the use or consideration of additional information in a weight-of-evidence assessment. "],
    "4.3": ["Outcome", "Report the interpretation of the model result in relation to the defined regulatory purpose."],
    "4.4": ["Conclusion", "Provide  an assessment of whether the final result is considered adequate for a regulatory \
    conclusion, or whether additional information is required (and, if so, what this additional information should be)."]
    }


    substance, prediction, nanoparticles, datapoints, qprf_datapoints, variables, variable_names, doaALL, qprf_creator = getJsonContentsQPRF(request.json)
    #print datapoints, len(datapoints)
    #print qprf_datapoints, len(qprf_datapoints)
    ens_sorted, pcafig_encoded = distances ([qprf_datapoints], datapoints, variables, [substance["uri"]], nanoparticles)
    #ens_sorted = [[[1]]]
    #pcafig_encoded = ""
    nearest = ""
    if len(ens_sorted[0][0]) >3:
        nearest = str(ens_sorted[0][0][0]) + ", " + str(ens_sorted[0][0][1]) + ", " + str(ens_sorted[0][0][2])
    else:
        for i in range (len(ens_sorted[0][0])):
            nearest += str(ens_sorted[0][0][i])

    if substance["cas"] !="":
        substance_dict["1.1"][1] = substance["cas"]
    if substance["ec"] !="":
        substance_dict["1.2"][1] = substance["ec"]
    if substance["iupac"] !="":
        substance_dict["1.3"][1] = substance["iupac"]
    if substance["inchi"] !="":
        substance_dict["1.5 b."][1] = substance["inchi"]

    general_dict["2.1"][1] = time.strftime("%d/%m/%Y")
    if qprf_creator != "":
        general_dict["2.2"][1] = qprf_creator

    if prediction["name"] !="":
        prediction_dict["3.1 a."][1] = prediction["name"]
        prediction_dict["3.1 b."][1] = prediction["name"]
    if prediction["model"] !="":
        prediction_dict["3.2 a."][1] = prediction["model"]
    if prediction["value"] !="":
        prediction_dict["3.2 d."][1] = prediction["value"]

    descriptor_string = ""
    for i in range (len (variable_names)):
        descriptor_string += str(variable_names[i]) + " = " + str(qprf_datapoints[i]) + ", "
    if descriptor_string !="":
        prediction_dict["3.2 g."][1] = descriptor_string

    if doaALL[0] != "" and doaALL[0] != None and doaALL[1] !="" and doaALL[1] != None:
        prediction_dict["3.3 a."][1] = "Value: " + str(doaALL[0]) + " for method: " + str(doaALL[1]) + ". Also, please see PCA figure included in this document."

    if nearest != "":
        prediction_dict["3.3 b."][1] = nearest

    task = OrderedDict()
    task = {
        "singleCalculations": {
                               "Title" : "QSAR Prediction Reporting Format (QPRF)",
                               "Version" : 1,
                               "Date" : time.strftime("%d/%m/%Y"),
                               "Time" : time.strftime("%H:%M:%S"),
                               "Disclaimer and Instructions" : "Please fill in the fields of the QPRF with information about the prediction and the substance \
                               for which the prediction is made. The information that you provide will be used to facilitate \
                               considerations on the adequacy of the prediction (model result) in relation to a defined \
                               regulatory purpose. \
                               The adequacy of a prediction depends on the following conditions: a) the (Q)SAR model is \
                               scientifically valid: the scientific validity is established according to the OECD principles for \
                               (Q)SAR validation; b) the (Q)SAR model is applicable to the query chemical: a (Q)SAR is \
                               applicable if the query chemical falls within the defined applicability domain of the model; c) \
                               the (Q)SAR result is reliable: a valid (Q)SAR that is applied to a chemical falling within its \
                               applicability domain provides a reliable result; d) the (Q)SAR model is relevant for the \
                               regulatory purpose: the predicted endpoint can be used directly or following an \
                               extrapolation, possibly in combination with other information, for a particular regulatory \
                               purpose. \
                               A (Q)SAR prediction (model result) may be considered adequate if it is reliable and relevant, \
                               and depending on the totality of information available in a weight-of-evidence assessment \
                               (see Section 4 of the QPRF)."
                              },
        "arrayCalculations": {
                               "1. Substance":
                                   {"colNames": ["Title", "Value"],
                                    "values": substance_dict
                                   },
                               "2. General information":
                                   {"colNames": ["Title", "Value"],
                                    "values": general_dict
                                   },
                               "3. Prediction":
                                   {"colNames": ["Title", "Value"],
                                    "values": prediction_dict
                                   },
                               "4. Adequacy (Optional)":
                                   {"colNames": ["Title", "Value"],
                                    "values": adequacy_dict
                                   }
                             },
        "figures": {
                   "PCA of Query instance vs. Training Dataset" : pcafig_encoded
                   }
        }
    """
    single = OrderedDict({     "Title" : "QSAR Prediction Reporting Format (QPRF)",
                               "Version" : 1,
                               "Date" : time.strftime("%d/%m/%Y"),
                               "Time" : time.strftime("%H:%M:%S"),
                               "Disclaimer and Instructions" : "Please fill in the fields of the QPRF with information about the prediction and the substance \
                               for which the prediction is made. The information that you provide will be used to facilitate \
                               considerations on the adequacy of the prediction (model result) in relation to a defined \
                               regulatory purpose. \
                               The adequacy of a prediction depends on the following conditions: a) the (Q)SAR model is \
                               scientifically valid: the scientific validity is established according to the OECD principles for \
                               (Q)SAR validation; b) the (Q)SAR model is applicable to the query chemical: a (Q)SAR is \
                               applicable if the query chemical falls within the defined applicability domain of the model; c) \
                               the (Q)SAR result is reliable: a valid (Q)SAR that is applied to a chemical falling within its \
                               applicability domain provides a reliable result; d) the (Q)SAR model is relevant for the \
                               regulatory purpose: the predicted endpoint can be used directly or following an \
                               extrapolation, possibly in combination with other information, for a particular regulatory \
                               purpose. \
                               A (Q)SAR prediction (model result) may be considered adequate if it is reliable and relevant, \
                               and depending on the totality of information available in a weight-of-evidence assessment \
                               (see Section 4 of the QPRF)."
                              })
    multip = OrderedDict({     "1. Substance":
                                   {"colNames": ["Title", "Value"],
                                    "values": substance_dict
                                   },
                               "2. General information":
                                   {"colNames": ["Title", "Value"],
                                    "values": general_dict
                                   },
                               "3. Prediction":
                                   {"colNames": ["Title", "Value"],
                                    "values": prediction_dict
                                   },
                               "4. Adequacy (Optional)":
                                   {"colNames": ["Title", "Value"],
                                    "values": adequacy_dict
                                   }
                             })
    figure = OrderedDict({
                   "PCA of Query instance vs. Training Dataset" : pcafig_encoded
                   })
    task = OrderedDict({
        "singleCalculations": single,
        "arrayCalculations": multip,
        "figures": figure
        })
    """
    #fff = open("C:/Python27/delete123.txt", "w")
    #fff.writelines(str(task))
    #fff.close 
    #task = {}
    jsonOutput = jsonify( OrderedDict(task) )
    
    return jsonOutput, 201 

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port = 5000, debug = True)

# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/qprf.json http://localhost:5000/pws/qprf
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/qprf3.json http://localhost:5000/pws/qprf
# C:\Python27\Flask-0.10.1\python-api 
# C:/Python27/python qprf.py
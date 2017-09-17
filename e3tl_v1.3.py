
# coding: utf-8

# In[35]:


"""
Spyder Editor

This is a temporary script file.
"""
import json
import re
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pprint import pprint
from IPython.display import Image, display, HTML
import ipywidgets as widgets
from ipywidgets import interactive


# In[36]:

Image("img/transform_detail.png")
logfn = "e3tl_execution_" + "{:%d%m%Y_%H%M%S}".format(datetime.now())
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename=logfn + '.txt', level=logging.DEBUG)

ALSFRS_Input = 'Tabela_Geral_09_2011_ALS-FRS.csv'
ALSFRS_SplitRules_JSON = 'ALSFRS_SplitRules.json'
ALSFRS_SemanticRules_JSON = 'ALSFRS_SemanticRules.json'


Demographics_Input = 'Tabela_Geral_09_2011_Demographics.csv'
Demographics_SplitRules_JSON = 'Demographics_SplitRules.json'
Demographics_SemanticRules_JSON = 'Demographics_SemanticRules.json'

JoinRules_JSON = 'JoinRules.json' 
data_dir = '..\\..\\'

fields_columns = ['Input File', 'Output File', 'Input Fields', 'Output Fields', 'Input Rows', 'Output Rows']
fields_df = pd.DataFrame(data=None, columns=fields_columns)
semantic_columns = ['Input File', 'Output File', 'Input Fields', 'Output Fields', 'Input Rows', 'Output Rows']
semantic_df = pd.DataFrame(data=None, columns=semantic_columns)
validation_columns = ['Input File', 'Field', 'Validation', 'Ok', 'Error']
validation_df = pd.DataFrame(data=None, columns=validation_columns)


# In[37]:

#Create a new function:
def num_missing(x):
    return sum(x.isnull())

def missingValues(data):
    #Applying per column:
    column_mv = data.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
    #Applying per row:
    row_mv = data.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row
    return row_mv, column_mv


def formatDateTime(datestr, outformat):    
    DATE_FORMATS = ['%m/%d/%Y %I:%M:%S %p', '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y', '%Y/%m/%d']
    my_date = None
    error = False
    for date_format in DATE_FORMATS:
        try:
            my_date = datetime.strptime(datestr, date_format).strftime(outformat)
            error = False
        except ValueError:
            error = True
            pass
        except TypeError:
            error = True
            break
        else:
          break
    return my_date, error

def displayTransparencyReport():
    global fields_df
    global entries_df
    global validation_df
    display(HTML('<h1>E3TL Transparency Summary</h1>'))
    display(HTML('<h2>Row Split</h2>'))
    display(fields_df)
    display(HTML('<h2>Semantic Validation</h2>'))
    display(semantic_df)
    display(HTML('<h2>Data Join</h2>'))
    display(validation_df)


# In[38]:

def RowSplit(in_data_fn, in_aux_file):
    in_file = data_dir + in_data_fn
    in_df = pd.read_csv(in_file, encoding='iso-8859-1', header=0)
    logging.info('Input file [%s] shape Rows:%d Columns:%d' % (in_file, in_df.shape[0], in_df.shape[1]))
    with open(in_aux_file) as data_file:    
        aux_json = json.load(data_file)
    exam_features = aux_json['ExamFeatures']
    exam_count_per_row = aux_json['ExamCountPerRow']
    out_features = aux_json['OutputFeatures']
    col_names=[]
    for feature in aux_json['FeatureMapping']:
        col_names.append(feature['Name'])    
    out_df = pd.DataFrame(columns=col_names)
    res_list = []
    exam_cnt = 0
    patient_cnt = 0

    validation_errors = {}
    validation_errors['Date'] = [0, 0]
    
    for idx,row in in_df.iterrows():
        patient_cnt += 1
        for exam in range(exam_count_per_row):
            out_row = [0]*out_features
            for feature in aux_json['FeatureMapping']:
                if (feature['StaticIdx'] == True):
                    out_row[feature['OutputIx']] = row[feature['InputIdx']]
                else:
                    out_row[feature['OutputIx']] = row[(exam*exam_features)+feature['InputIdx']]
                if ('Type' in feature and feature['Type'] == 'Date'):
                    res, error = formatDateTime(out_row[feature['OutputIx']], feature['Format']) 
                    out_row[feature['OutputIx']] = res
                    if (error):
                        validation_errors['Date'][1] = validation_errors['Date'][1] + 1
                    else:
                        validation_errors['Date'][0] = validation_errors['Date'][0] + 1
            exam_cnt += 1
            res_list.append(out_row)
    out_df = pd.DataFrame(res_list, columns=col_names)
    logging.info('Field Mapping output file shape Rows:%d Columns:%d' % (out_df.shape[0], out_df.shape[1]))
    out_df.to_csv(aux_json['OutFile'], index=False)
    
    #report information
    global fields_df
    fields_df = fields_df.append(pd.DataFrame([[in_file, aux_json['OutFile'], in_df.shape[1], out_df.shape[1], in_df.shape[0], out_df.shape[0]]], columns=fields_columns), ignore_index = True)
#   display(fields_df)    
    global validation_df    
    validation_df = validation_df.append(pd.DataFrame([[in_file, 'Date', 'Format', validation_errors['Date'][0], validation_errors['Date'][1]]], columns=validation_columns), ignore_index = True)
    #display(validation_df)   
    return aux_json['OutFile']


# In[39]:

def SemanticValidation(in_data_fn, in_aux_file):  
    with open(in_aux_file) as data_file:    
        aux_json = json.load(data_file)
    interm_data = pd.read_csv(in_data_fn, encoding='iso-8859-1', header=0)
    logging.info('Appyling rules to intermiary file %s with %d rows and %d columns' % (in_data_fn, interm_data.shape[0], interm_data.shape[1]))    
    out_df = pd.DataFrame(data=None, columns=interm_data.columns)
    logging.info('For each input file row...apply Validation Rules')
    
    validation_errors = {}
   
    for in_idx,row in interm_data.iterrows():
        valid_row = True
        for rule in aux_json['SemanticRules']:
            idx = rule['InputIdx']
            field_name = interm_data.columns[idx]
            validation_errors[field_name] = validation_errors.get(field_name, {'Threshold':[0,0], 'RegEx':[0,0], 'NotNull':[0,0]})
            if ('NotNull' in rule and rule['NotNull'] is True):
                v_res = validation_errors[field_name]['NotNull']
                if (pd.isnull(row[idx])):
                    valid_row = False
                    v_res = [v_res[0], v_res[1] + 1]
                else:
                    v_res = [v_res[0]+1, v_res[1]]
                validation_errors[field_name]['NotNull']= v_res
            if ('Threshold' in rule):
                threshold_sign = rule['Threshold'][0]
                threshold_value = float(rule['Threshold'][1:])
                v_res = validation_errors[field_name]['Threshold']
                try:
                    row_value = float(row[idx])
                    if (threshold_sign == '<' and row_value >= threshold_value):
                        valid_row = False
                        v_res = [v_res[0], v_res[1] + 1]
                    elif (threshold_sign == '>' and row_value <= threshold_value):                        
                        valid_row = False
                        v_res = [v_res[0], v_res[1] + 1]
                    else:
                        v_res = [v_res[0]+1, v_res[1]]
                except ValueError:
                    valid_row = False
                validation_errors[field_name]['Threshold']= v_res
            if ('RegEx' in rule):
                v_res = validation_errors[field_name]['RegEx']
                try:
                    test_regex = re.compile(rule['RegEx'])
                    if (pd.isnull(row[idx]) or test_regex.match(row[idx]) is None):
                        valid_row = False
                        v_res = [v_res[0], v_res[1] + 1]
                    else:
                        v_res = [v_res[0]+1, v_res[1]]
                except ValueError:
                    valid_row = False
                validation_errors[field_name]['RegEx'] = v_res
        if (valid_row):
            out_df = out_df.append(row)
    logging.info('Output format:%d rows x %d columns' % (out_df.shape[0], out_df.shape[1]))
    out_df.to_csv(aux_json['OutFile'], index=False)
    
    #report information
    global semantic_df
    semantic_df = semantic_df.append(pd.DataFrame([[in_data_fn, aux_json['OutFile'], interm_data.shape[1], out_df.shape[1], interm_data.shape[0], out_df.shape[0]]], columns=semantic_columns), ignore_index = True)
    #display(semantic_df)
    global validation_df    
    temp_validation_df = pd.DataFrame.from_dict(validation_errors)
    temp_validation_df['File'] = in_data_fn
    #display(temp_validation_df)
    #validation_df = validation_df.append(temp_validation_df, ignore_index = True)
    #display(validation_df)
    


# In[40]:

def DataJoin(rules):
    with open(rules) as data_file:    
        aux_json = json.load(data_file)
#    interm_data = pd.read_csv(intermediate_fn, encoding='iso-8859-1', header=0)
#    print ('Appyling rules to intermiary file %s with %d rows and %d columns' % (intermediate_fn, interm_data.shape[0], interm_data.shape[1]))    
    out_df = pd.DataFrame(data=None)    
    keys = []
    in_dfs = {}
    for rule in aux_json['FeatureMapping']:
        logging.info (rule)
        if (rule["InputFile"] not in in_dfs):
            logging.info('Loading DataFrame from:' + rule["InputFile"])
            in_dfs[rule["InputFile"]] = pd.read_csv(rule["InputFile"], encoding='iso-8859-1', header=0)    
        if ('JoinKey' in rule and rule['JoinKey'] is True):
            keys.extend([rule['Name']])
    logging.info('Columns to be used as Keys:' + str(keys))
    in_dfs = [ v for v in in_dfs.values() ]
    out_df = in_dfs[0].join(in_dfs[1], on=keys, how='inner', rsuffix='r_')
    out_df.to_csv(aux_json['OutFile'], index=False)


# <h2>1:m Row Split</h2>

# In[41]:

intermediary_data_fn = RowSplit(ALSFRS_Input, ALSFRS_SplitRules_JSON)


# In[42]:

intermediary_data_fn2 = RowSplit(Demographics_Input, Demographics_SplitRules_JSON)


# <h2>Semantic Validation</h2>

# In[43]:

SemanticValidation(intermediary_data_fn, ALSFRS_SemanticRules_JSON)


# In[44]:

SemanticValidation(intermediary_data_fn2, Demographics_SemanticRules_JSON)


# <h2>Data Join</h2>

# In[45]:

DataJoin(JoinRules_JSON)


# In[46]:

displayTransparencyReport()


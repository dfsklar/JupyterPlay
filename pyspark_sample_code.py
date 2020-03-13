from pyspark.sql.functions import *
from pyspark.sql.functions import col
import numpy as np
import re
from pyspark.sql import Window
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import *
import numpy as np


#-------------------------------------------------------------------------------------#
# Identifying variable types                                                          #
#-------------------------------------------------------------------------------------#

def variable_types(df):
	ex_var=['account','accountid','account_number','housekey','customer_account_id','region','division','accountid','target']
	vars=df.dtypes
	categorical=[]
	numeric=[]
	for i in xrange(0,len(vars)):
		if vars[i][1]=="string" and vars[i][0] not in ex_var :
			categorical.append(vars[i][0])
		elif vars[i][1] not in ['timestamp','date'] and vars[i][0] not in ex_var:
			numeric.append(vars[i][0])
		
	return numeric, categorical


#-------------------------------------------------------------------------------------#
# Convert fake categorical back numeric                                               #
#-------------------------------------------------------------------------------------#
def cat_to_numeric(df,categorical):
	import re
	tonumeric_list=[]
	for x in categorical:
        #if (re.search("_pct",x)) or (re.search("dayssince",x)) or (re.search("count",x)) or (re.search("age",x)) or (re.search("cnt",x)) or (re.search("min",x)) or (re.search("vi_",x)) or (re.search("amt",x)) :
		if df.agg(avg(x).alias('avg')).collect()[0]['avg'] is not None:
			tonumeric_list.append(x)
	for var in tonumeric_list:
		df=df.withColumn(var,df[var].cast("Double"))
	return df,tonumeric_list


#-------------------------------------------------------------------------------------#
# Filter out variables not qualified for modeling                                     #
#-------------------------------------------------------------------------------------#


def exclude_vars(df,custom_exclude):
	
	date_vars=[]
	for x in df.columns:
		if (re.search("_dt",x)) :
			date_vars.append(x)
	
	key_vars=[]
	for x in df.columns:
		if (re.search("key",x)) or (re.search("sysprin",x)):
			key_vars.append(x)
	
	model_vars=[]
	for x in df.columns:
		if (re.search("score",x)) or (re.search("segment",x)):
			model_vars.append(x)
	
	location_vars=[]
	for x in df.columns:
		if (re.search("_stat_abbr",x)) or (re.search("serloc_",x)) or (re.search("state",x)) or (re.search("region",x)) or (re.search("division",x)) or (re.search("dma",x)) or (re.search("zip",x)) or (re.search("entity",x)) or (re.search("longtitude",x)) or (re.search("latitude",x)):
			location_vars.append(x)
	
	bus_vars=[]
	for x in df.columns:
		if (re.search("_bus_",x)) or (re.search("epsilon_",x)):
			bus_vars.append(x)
	
	
	future_vars=[]
	for x in df.columns:
		if (re.search("nxtmo",x)):
			future_vars.append(x)
	        
	additional=[]
	for x in df.columns:
	    for a in custom_exclude:
	        if (re.search(a,x)):
	            additional.append(x)
	
	excludevars=date_vars+key_vars+model_vars+location_vars+bus_vars+future_vars+additional
	
	df=df.drop(*excludevars)
	
	return df,excludevars



#-------------------------------------------------------------------------------------#
# Missing value analysis                                                              #
#-------------------------------------------------------------------------------------#



def missing_analysis(df,features,numeric,categorical,threshold):
# Identify highly missing variabls 
	drop_var_list=[]
	all_miss_list=[]
	miss_list_numeric_withzero=[]
	miss_list_numeric_withnull=[]
	miss_list_categ=[]
	miss_expdemo=[]
	total=df.count()
	for var in list(set(features)-set(['target','accountid','customer_account_id'])):
		null_count=df.where((col(var).isNull())|(col(var)=='NULL')).count()
		pct_null=null_count/float(total)
		
		if ("expdemo_" in var) & (var in numeric):
			miss_expdemo.append(var)
		elif pct_null==1:
			drop_var_list.append(var)
		elif (pct_null>=threshold) & (var in numeric):
			zero_cnt=df.where(col(var)==0).count()
			if zero_cnt>0:
				miss_list_numeric_withzero.append(var)
			else:
				drop_var_list.append(var)
		elif (pct_null>=threshold) & (var in categorical):
			drop_var_list.append(var)
		elif (pct_null<threshold) :
			if var in numeric:
				miss_list_numeric_withnull.append(var)
			elif var in categorical:
				miss_list_categ.append(var)
	return all_miss_list,miss_list_numeric_withzero,miss_list_numeric_withnull,miss_list_categ,miss_expdemo,drop_var_list



#-------------------------------------------------------------------------------------#
# Imputing missing values based on pattern and logic                                  #
#-------------------------------------------------------------------------------------#
                
def imputation(df,numeric,miss_list_numeric_withzero,miss_list_numeric_withnull,miss_list_categ,miss_expdemo,drop_var_list):
	median={}
	df2=df.drop(*drop_var_list)
	
	for var in miss_list_numeric_withzero:
		df2=df2.na.fill(-1,subset=var)
	 
	for var in miss_expdemo:
		medianValue=df2.approxQuantile(var,[0.5],0.00001)[0]
		median[var]=medianValue
		df2=df2.na.fill(medianValue,subset=[var])
	    
	for var in miss_list_numeric_withnull:
		df2=df2.na.fill(0,subset=[var])
	
	for var in miss_list_categ:
		df2=df2.na.fill('OTHER',subset=[var])
	    
	return df2,median


#-------------------------------------------------------------------------------------#
# Identify highly correlated variables                                                #
#-------------------------------------------------------------------------------------#


def corr_drop(df,features,threshold):
    numeric, categorical=variable_types(df.select(*list(set(features) & set(df.columns))))
    numeric_new=list(set(numeric)-set(['account_number','customer_account_id','account','accountid','target']))
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=numeric_new, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    corrmatrix = matrix.toArray().tolist()
    #format to pair 
    datalist=[]
    for i in range(len(numeric)):
        for j in range(i+1, len(numeric)):
            datalist.extend([[numeric[i],numeric[j],corrmatrix[i][j]]])
    dfcorr = spark.createDataFrame(datalist,['var1','var2','corr'])
    #dfcorr=dfcorr.withColumn('adj_corr',abs(col('corr')))
    # Find index of feature columns with correlation greater than threshold
    to_drop = [var[0] for var in dfcorr.where(abs(col('corr'))>threshold).select('var2').collect()]
    fea_cols=list(set(features)-set(to_drop))
    return fea_cols, to_drop


#-------------------------------------------------------------------------------------#
# Identify variables with high cardinality                                            #
#-------------------------------------------------------------------------------------#

def high_cardinality(df,categorical):
    cardin_dict={}
    for col in categorical:
        c=df.select(col).distinct().count()
        if c>50:   
            df_temp=df.groupBy(col)\
            .agg(count(col).alias('cnt'),avg('target').alias('pred_rate'))\
            .orderBy(['cnt','pred_rate'],ascending=False).limit(20)
	
            keep_list=[x[0] for x in df_temp.select(col).collect()]
	
            df=df.withColumn(col,when(~df[col].isin(keep_list),'OTHER_CAT').otherwise(df[col]))
            cardin_dict[col]=[x[0] for x in df.select(col).distinct().collect()]
            
    return df, cardin_dict
        

#-------------------------------------------------------------------------------------#
# Variable treatment main functions                                                   #
#-------------------------------------------------------------------------------------#


def var_treatment(df,features,threshold=0.7,corr_threshold=0.8,custom_exclude=''):
	df=df.select(*features)
	numeric,categorical=variable_types(df)
	
	df1,tonumeric_list=cat_to_numeric(df,categorical)
	
	print('cat_to_numeric')
	df1,excludevars=exclude_vars(df1,custom_exclude)
	print('exclude')
	df1=df1.drop(*excludevars)
	print('drop')
	numeric,categorical=variable_types(df1)
	all_miss_list,miss_list_numeric_withzero,miss_list_numeric_withnull,miss_list_categ,miss_expdemo,drop_var_list= missing_analysis(df1,features,numeric,categorical,threshold)
	print('missing')
	df1,median=imputation(df1,numeric,miss_list_numeric_withzero,miss_list_numeric_withnull,miss_list_categ,miss_expdemo,drop_var_list)
	print('impute')
	fea_cols, to_drop=corr_drop(df1,features,corr_threshold)
	print('correlation')
	df1=df1.drop(*to_drop)
	
	numeric,categorical=variable_types(df1)
	
	dfx,cardin_dict =high_cardinality(df1,categorical)
	print('high car')
    
	print('Highly correlated variabls to drop - ', to_drop)
	print('Variables not qualified for modeling - ',excludevars)
	print('High cardinality variables  - ',cardin_dict)
	print('Categorical variables converted to numeric - ', tonumeric_list)
	print('Variables dropped - ',drop_var_list)
	print('Numeric variables imputed to -1 - ',miss_list_numeric_withzero)
	print('Numeric variables imputed to median - ',median)
	print('Numeric variables imputed to 0 - ',miss_list_numeric_withnull)
	print('Categorical variables imputed to "OTHER" - ',miss_list_categ)
	
	return dfx





#-------------------------------------------------------------------------------------#
# Gradient Boosting Trees Model Training                                              #
#-------------------------------------------------------------------------------------#




from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
import numpy as np


def gbt_train(inputdata,maxiter,maxdep,maxBins,seed):
	
	ex_var=['account','customer_account_id','housekey','account_number','accountid','target','division','region','decile_segment_flag','year','month','athena_business_id']
	vars=inputdata.dtypes
	categorical=[]
	numeric=[]
	for i in xrange(0,len(vars)):
		if vars[i][1]=="string" and vars[i][0] not in ex_var:
			categorical.append(vars[i][0])
		elif vars[i][1] not in ['timestamp','date'] and vars[i][0] not in ex_var:
			numeric.append(vars[i][0])
	
	
	#Update the list of categorical vars after indexing (for features building)
	categorical_index=[]
	for i in xrange(0,len(categorical)):
		categoricals=categorical[i]+'_index'
		categorical_index.append(categoricals)
	
	
	#Convert categorical variables to numeric
	stages = [] # stages in our Pipeline
	for categoricalCol in categorical:
	  # Category Indexing with StringIndexer
	  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+'_index').setHandleInvalid("keep")
	  stages += [stringIndexer]
	
	
	assembler = VectorAssembler(inputCols=categorical_index+numeric, outputCol='features')
	stages+=[assembler]
	
	#labelIndexer = StringIndexer(inputCol='target', outputCol="label")
	#stages+=[labelIndexer]
	
	gbt = GBTClassifier(labelCol="target", featuresCol="features", maxIter=maxiter,maxDepth=maxdep,seed=seed,maxBins=maxBins)
	stages+=[gbt]
	
	pipeline = Pipeline(stages=stages)
	
	 
	pipelineModel = pipeline.fit(inputdata)
	
	
	return pipelineModel




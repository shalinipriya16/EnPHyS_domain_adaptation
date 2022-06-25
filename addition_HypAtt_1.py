import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval as le
from keras import Input, Model#, objectives
import keras

from HYP_1 import advOrthoMTL 
from POS_1 import Pos


np.random.seed(0)

modelname="repo/resOutput/HYP/completemodel_HYPAtt.h5"

pin="../../../xyData/xyData/embeddedFiles/"
pout="repo/resOutput/HYP_ItHuInNe_tsne/"#ItHu/"
if not os.path.exists(pout):
	os.makedirs(pout)
srcL=["Italy2","Hurricane2","Indonesia2" ]# ["Hurricane2_120"]#, "Nepal2"]#, "Nepal2"]#, "Italy2"]#, "Nepal2"]#, "Hurricane2"]
tar= "Nepal2"#"Indonesia2_120"#"Indonesia2" #"Nepal2" #"Indonesia2"#"Italy2", "Nepal2"
p=0.8
Epochs=500#665#5000
partN="_".join(srcL); partN=partN+"_"+tar
statsFile=pout+"dfStats_HYP_"+partN+"_"+str(Epochs)+".csv"
predictedFile=pout+"pred_"+tar+"_"+str(Epochs)+".pkl"

resFile=pout+"resDf_HYP_"+partN+"_"+str(Epochs)+".csv"

def getData():
	dfL=[]
	N=len(srcL)+1
	for i in range(len(srcL)):
		#read
		dtemp=pd.read_csv(pin+srcL[i]+".csv")
		#append into list
		dfL.append(dtemp)
	dfL.append(pd.read_csv(pin+tar+".csv"))
	#print (dfL); input()
	for i in range(len(dfL)):
		#dtemp=dfL[i]
		dfL[i]["label"]=dfL[i]["label"].apply(lambda x:int(x))
		dfL[i]["padded_sentences"]=dfL[i]["padded_sentences"].apply(lambda x:le(x))
		dfL[i]["pos_embed"]=dfL[i]["pos_embed"].apply(lambda x:le(x))

	dfL=[ x[["padded_sentences", "pos_embed", "text", "label"]] for x in dfL ]
	#print (dfL); input()
	def labelCount(df):
		return (df["label"].tolist().count(0), df["label"].tolist().count(1))

	countL=[ labelCount(x) for x in dfL]
	countL[-1]=(countL[-1][0], int(countL[-1][1]/2))
	print (countL)
	max1=max([x[1] for x in countL])#; max1=int(0.8*max1)

	def getSamplesBeforeTraining(df, maxN, i):
		df1=df[ ( (df["label"]==1) ) ]
		t=df1.shape[0]
		df0=df[ ( (df["label"]==0) ) ]
		f=df0.shape[0]
		print ("values ", t, f, maxN, i)
		print ("ones and zeros in data ", t, f, maxN)
		#etr, etv, ets=-1, -1, -1
		if i==N-1:
			trT=int(0.5*t); tsT=t-trT; vlT=int(0.1*trT); trT=trT-vlT
			etr=pd.concat([ df1[:trT], df0[:maxN-trT] ])
			etv=pd.concat([ df1[trT:trT+vlT], df0[maxN-trT:2*maxN-trT-vlT] ])
			ets=pd.concat([ df1[trT+vlT:trT+vlT+tsT], df0[2*maxN-trT-vlT: 3*maxN-trT-vlT-tsT] ])
			print ("one and zeros inside tar  ", df1[trT+vlT:trT+vlT+tsT].shape, df1[trT+vlT:trT+vlT+tsT][ ( (df1["label"]==1) ) ].shape, etr.shape, etv.shape, ets.shape)
		else:
			trT=t; tsT=t-trT; vlT=int(0.1*trT); trT=trT-vlT
			etr=pd.concat([ df1[:trT], df0[:maxN-trT] ])
			etv=pd.concat([ df1[trT:trT+vlT], df0[maxN-trT:2*maxN-trT-vlT] ])
			ets=pd.concat([ df0[2*maxN-trT-vlT: 3*maxN-trT-vlT-tsT] ])
			print ("one and zeros inside  src  ", etr.shape, etv.shape, ets.shape)

		for i in range(100):
			etr=etr.sample(frac=1, random_state=1)
			#etv=etv.sample(frac=1)
			
		return etr, etv, ets
		
	ratio=2	
	maxN=max1*ratio	
	tr, tv, ts=[], [], []
	for i in range(len(dfL)):
		etr, etv, ets=getSamplesBeforeTraining(dfL[i], maxN, i)
		tr.append(etr); tv.append(etv); ts.append(ets)
	
	trs=tr[0].shape[0]; tvs=tv[0].shape[0]; tss=ts[0].shape[0]
	
	mtl_tr=[ [df["label"].tolist()[i] for df in tr] for i in range(trs) ]
	mtl_tv=[ [df["label"].tolist()[i] for df in tv] for i in range(tvs) ]
	mtl_ts=[ [df["label"].tolist()[i] for df in ts] for i in range(tss) ]	
	
	xtr=[ x["padded_sentences"].tolist() for x in tr ]; ytr=[ x["label"].tolist() for x in tr ]+[mtl_tr]+[ x["pos_embed"].tolist() for x in tr ]
	xtv=[ x["padded_sentences"].tolist() for x in tv ]; ytv=[ x["label"].tolist() for x in tv ]+[mtl_tv]+[ x["pos_embed"].tolist() for x in tv ]
	xts=[ x["padded_sentences"].tolist() for x in ts ]; yts=[ x["label"].tolist() for x in ts ]+[mtl_ts]+[ x["pos_embed"].tolist() for x in ts ]
	
	##for text
	xtr_text=[ x["text"].tolist() for x in tr ]; xtv_text=[ x["text"].tolist() for x in tv ]; xts_text=[ x["text"].tolist() for x in ts ]
	
	def replaceWithNumpy(lis):
			n=len(lis)
			e=1
			try:
				e=len(lis[0])
			except:
				#print (type(lis[0]), lis[0])
				e=1
			a=np.zeros((n, e))
			for i in range(len(lis)):
				a[i, :]=lis[i]
			return a

	xtr=[replaceWithNumpy(x) for x in xtr]; ytr=[replaceWithNumpy(x) for x in ytr]
	xtv=[replaceWithNumpy(x) for x in xtv]; ytv=[replaceWithNumpy(x) for x in ytv]
	xts=[replaceWithNumpy(x) for x in xts]; yts=[replaceWithNumpy(x) for x in yts]
	
	#print ("trailing ", yts[N-1][:5], yts[N-1][-5:])
	trss=xtr[0].shape[0]
	xtv=[x[:int(0.1*trss)] for x in xtv]; ytv=[x[:int(0.1*trss)] for x in ytv]
	xts=[x[:2*countL[-1][1]] for x in xts]; yts=[x[:2*countL[-1][1]] for x in yts]; xts_text=[x[:2*countL[-1][1]] for x in xts_text]
	#print ("trailing 2 ", yts[N-1][:5], yts[N-1][-5:])
	
	#print ("every tr tv ts ", [x.shape for x in xtr], [x.shape for x in xtv], [x.shape for x in xts], max1, 2*int(0.2*max1), int(0.1*trss))
	
	print ("XTR AND Xts initially ")
	print ([x.shape for x in xtr])
	print ([x.shape for x in ytr])
	print ([x.shape for x in xtv])
	print ([x.shape for x in ytv])
	print ([x.shape for x in xts])
	print ([x.shape for x in yts])
	
	bt=xtr[0].shape[0]%32
	if bt!=0:
		xtr=[x[:-bt] for x in xtr]; ytr=[x[:-bt] for x in ytr]
		xtr_text=[x[:-bt] for x in xtr]##text
		
	bt=xtv[0].shape[0]%32
	if bt!=0:
		xtv=[x[:-bt] for x in xtv]; ytv=[x[:-bt] for x in ytv]
		xtv_text=[x[:-bt] for x in xtv]##text
	
	bt=xts[0].shape[0]%32
	if bt!=0:
		xts=[x[:-bt] for x in xts]; yts=[x[:-bt] for x in yts]
		xts=[xts[-1] for x in xts]
		yts=[ yts[N-1], yts[N] ]
		xts_text=[x[:-bt] for x in xts_text]

	print ("XTR AND Xts finally ")
	print ([x.shape for x in xtr])
	print ([x.shape for x in ytr])
	print ([x.shape for x in xtv])
	print ([x.shape for x in ytv])
	print ([x.shape for x in xts])
	print ([x.shape for x in yts])
	print ([len(x) for x in xts_text])#; input("enter a number")
	
	return xtr, ytr, xtv, ytv, xts, yts, xtr_text, xtv_text, xts_text

#getData()		

def trainTest(**kwargs):

	global srcL; global Epochs	

	maxLength=80; batch_size=32; N=len(srcL)+1
	
	#ps=Pos()
	
	#psTup=ps.posModel4Grp(srcL); psm=psTup[-1]

	xtr, ytr, xtv, ytv, xts, yts, xtr_text, xtv_text, xts_text=getData()	
	
	trs=xtr[0].shape[0]; bt=batch_size
	#keras.backend.clear_session()
	###another model		
	
	kwargs={"N":N, "MAX_SEQUENCE_LENGTH":maxLength, "vocab_size":20000, "output_word_dim":200, "input_sequence_length":150, "no_of_filters":5, "filter_size":5, "maxPool_size":5, "dropout_fraction":0.5, "lstm_output_dim":150, "dense_dim_1":150, "dense_dim_2":40}
	
	model1 = advOrthoMTL(**kwargs)
		
	##data for ensemble_addition model
	xtr_ensemble=xtr[:N-1]+[ xtr[N-1] for i in range(N)] ; ytr_ensemble=ytr[:N-1]+[ytr[N-1] for i in range(N+1)]
	xts_ensemble=xts[:N-1]+[ xts[N-1] for i in range(N)]; yts_ensemble=[yts[0] for i in range(N+1)]
	xtv_ensemble=xtv[:N-1]+[ xtv[N-1] for i in range(N)] ; ytv_ensemble=ytv[:N-1]+[ytv[N-1] for i in range(N+1)]
	
	trs=xtr_ensemble[0].shape[0]; bt=batch_size
	his=model1.fit(xtr_ensemble, ytr_ensemble, epochs=Epochs, batch_size=batch_size, validation_data=(xtv_ensemble, ytv_ensemble), verbose=2)

	#print (his.history)
	dfOrthoStats=pd.DataFrame()
	dfOrthoStats["loss"]=his.history["loss"]; dfOrthoStats["val_loss"]=his.history["val_loss"] 
	dfOrthoStats["tarPred_accuracy"]=his.history["tarPred_accuracy"]
	dfOrthoStats["val_tarPred_accuracy"]=his.history["val_tarPred_accuracy"]
	dfOrthoStats["tarCustomPred_accuracy"]=his.history["tarCustomPred_accuracy"]
	dfOrthoStats["val_tarCustomPred_accuracy"]=his.history["val_tarCustomPred_accuracy"]
	dfOrthoStats.to_csv(statsFile, index=False)
		
	model1.save(modelname)
	target_Model=Model(model1.inputs, model1.outputs[N-1:])
	
	layersL=[ (target_Model.layers[i].name, i) for i in range(len(target_Model.layers)) ]
	print ("\n name, output shape of each ", [(y, x.output.shape) for y, x in zip(layersL, target_Model.layers)])
	#input("enter")
	
	y_Ortho=target_Model.predict(xts_ensemble)
	
	### check are we using correct test set 
	dfRes=pd.DataFrame()
	dfRes["text"]=xts_text[-1]
	dfRes["real"]=[x[0] for x in yts_ensemble[0].tolist()] 
	for i in range(N-1):
		dfRes["tar_"+srcL[i]]= [x[0] for x in y_Ortho[i].tolist()]		
	
	dfRes["tar"]= [x[0] for x in y_Ortho[N].tolist()] 
	dfRes["tar_custom"]= [x[0] for x in y_Ortho[-1].tolist()]	
	
	dfRes.to_csv(resFile, index=False)
	
	vecModel=Model(inputs=target_Model.inputs, outputs=[ target_Model.layers[32].output, target_Model.layers[28].output ])
	vec=vecModel.predict(xts_ensemble); print (" shape in vec ", [x.shape for x in vec])
	f=open(pout+"hyp_vecFile_"+str(Epochs)+".pkl", "wb")
	pickle.dump(vec, f)
	f.close()

	return


trainTest()



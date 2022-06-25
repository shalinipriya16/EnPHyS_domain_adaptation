import pandas as pd
import numpy as np
from ast import literal_eval as le

pt="repo/resOutput/HYP/"

def prog(resDf):
	df=pd.read_csv(resDf); N=2
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	print(df.columns)
	col=df.columns.tolist(); colN=col[-4:]
	
	for i, c in enumerate(colN):
		
		if c=="real" or c=="text":# or c=="Unnamed: 0_Y":
			continue
		#print (c); input("enter")
		lis=df[c].tolist()
		mn=np.mean(lis)
		std=np.std(lis)
		threshold=mn+0.9*std
		if i<N:
			threshold=0.09
			lis=[1 if x>=threshold else 0 for x in lis]
			df[c+"_Y"]=lis
		if c=="tar":
			threshold=0.09
			lis=[1 if x>= threshold else 0 for x in lis]
			df[c+"_Y"]=lis
		if c=="tar_custom":
			threshold=0.2
			lis=[1 if x>= threshold else 0 for x in lis]
			df[c+"_Y"]=lis
		if c=="pred":
			threshold=0.5
			lis=[1 if x>= threshold else 0 for x in lis]
			df[c+"_Y"]=lis

	#for i, c1 in enumerate(col):
	#	for j, c2 in enumerate(col):
	#		if i<j:
	#			print ("\n correlation between ", c1, "___", c2, "====", np.corrcoef(df[c1].tolist(), df[c2].tolist())[0][1], "\n")
	
	def conf(r, p):
		tp, tn, fp, fn=0, 0, 0, 0
		for x, y in zip(r, p):
			if x==y and x==1:tp+=1
			if x!=y and x==1:fn+=1
			if x==y and x==0:tn+=1
			if x!=y and x==0:fp+=1
		print ("tp tn fp fn ", tp, tn, fp, fn)
		pre, rec, f1=0, 0, 0
		try:
			pre=1.0*tp/(tp+fp)
			#pre=1.0*(tp+tn)/(tp+fp+tn+fn)
		except:
			print ("precision is set to 0 ")
			pre=0
		try:
			rec=1.0*tp/(tp+fn)
			#rec=1.0*(tp+tn)/(tp+fn+tn+fp)
		except:
			print ("recall is set to 0 ")
			rec=0
		try:
			f1=2.0*pre*rec/(pre+rec)
		except:
			print ("f1 is set to 0 ")
			f1=0

		print (" precision recall f1 ", pre, rec, f1)

		return pre, rec, f1

	metricL=[]
	for c in col:
		if c!="real" and c!="text":
			r, p=df["real"].tolist() ,df[c+"_Y"].tolist()
			print ("_______", c, "________")
			print ("real 1 0 pred 1 0 ___", r.count(1), r.count(0), p.count(1), p.count(0))			
			metricL.append(conf(r, p))

	print ("______agg______")
	r, p, mtl,pred=df["real"].tolist(), df["tar_Y"].tolist(), df["tar_custom_Y"].tolist(), df["pred_Y"].tolist()
	#lis=[1 if x==1 or y==1 or z==1 else 0 for x, y,z in zip(p, mtl, pred)]
	lis=[1 if x==1 or y==1 else 0 for x, y in zip(p, pred)]

	print ("real 1 0 pred 1 0 ___", r.count(1), r.count(0), lis.count(1), lis.count(0))
	conf(r, lis)
	df["agg"]=lis

	df.to_csv(pt+"newResDf.csv", index=False)
	return


files=pt+"resDf_HYP_Hurricane2_Nepal2_5000.csv"#"resDf_HYP_Hurricane2_Indonesia2_365.csv"#"resDf_HYP_Hurricane2_Nepal2_5000.csv"
ff="repo/resOutput/POS/resDf_pos_Hurricane2_Nepal2_200.csv"#"repo/resOutput/POS/resDf_pos_Hurricane2_Indonesia2_2000.csv"#"repo/resOutput/POS/resDf_pos_Hurricane2_Nepal2_200.csv"

dff=pd.read_csv(files)
#print(dff.columns)#; input("enter a number")
df2=pd.read_csv(ff)
#print(dff.columns)#; input("enter a number")
pred=df2["pred"].to_list()
pred_M= df2["pred_MTL"].to_list()

dff["pred"]=pred
#dff["pred_MTL"]=pred_M
#print(dff.columns); input("enter a number")
dff.to_csv("resDf.csv")
prog("resDf.csv")#("resDf200_1200.csv")#("resDf1500_2000.csv")
	






















		

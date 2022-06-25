import pickle

def prog(fl):
	f=open(fl, "rb")
	f.seek(0)
	var=pickle.load(f)
	f.close()
	print ("\n vec \n", var)
	print ("\n type \n", [(type(x), x.shape) for x in var])
	return var
	
	
if __name__=="__main__":
	fl="repo/resOutput/HYP_ItHuInNe_tsne/hyp_vecFile_500.pkl"
	prog(fl)		
	
	
	
	

#import Files and Packages
import numpy as np
import shutil
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import sys



##############################################################
####### INPUTS ###############################################
##############################################################


# Symmetric   -> Horn 3 [2]  (30,-304)
# Elliptical  -> Horn 1 [0] (510,-304) 
# Assymmetric -> Horn 28 [27] (-990,304)

# Indices das cornetas a serem analisadas:
horn_idxs = np.array([2])
Polarisation = ["linear_x"] #["linear_x","linear_y"]
Fs = np.linspace(0.98,1.26,1) # Frequencias em GHz

# Grid
Nps = [101] # Lista com numero de pontos a serem analisados
# The uv grid to be used will be uv_center+-d_uv, yielding a (2*d_uv)**2 grid:
d_uv = 0.05

jobNamePrefix ="Retangular"
# Path para a pasta que contem o arquivo dentro do usuario no local:
# GRASP-SE-XX.X.X/bin/grasp-analysis
realpath = "/home/joao/TICRA/GRASP-SE-10.3.0/bin/grasp-analysis batch.gxp "



##############################################################
####### FUNCOES ##############################################
##############################################################


def taper_angle(f):
	'''
	FIT TAPER ANGLE DO LIT PARA 12dB
	Taper angle (X==frequencia) = -1.46802646e2 + 1.10877204e-1*X + -2.96922720e-5*X^2 + 9.04792596e4/X + -1.19053839/X^2
	
	f input: GHz
	'''
	f = f*1000.
	return-1.46802646e2 + 1.10877204e-1*f + -2.96922720e-5*f**2 + 9.04792596e4/f + -1.19053839/f**2


def grd2txt(pathin,pathout,Np=922,index=None):
	'''
	copia os dados do spherical_grid.grd só com as informacoes dos campos para um arquivo spherical_grid.txt
	'''
	
	name = "spherical_grid"
	pathname = os.path.join(pathin,name)
	f = open(pathname+".grd","r")
	spherical_grid_text = f.read()
	f.close()
	tab    = spherical_grid_text.split("\n")
	wr = spherical_grid_text.split("\n	   "+str(int(Np))+"	    "+str(int(Np))+"	       0\n")[-1]
	pathname = os.path.join(pathout,name)
	f = open(pathname+"_"+str(int(index))+".txt","w+")
	f.write(wr)
	f.close()
	return None



##############################################################
####### DADOS ################################################
##############################################################


# Coordenadas de cada uma das 28 cornetas
# NÃO ALTERAR
xOr=[  510.,   270.,	30.,  -210.,  -450.,  -690.,  -930.,
	   390.,   150.,   -90.,  -330.,  -570.,  -810., -1050.,   
	   330.,    90.,  -150.,  -390.,  -630.,  -870., -1110.,   
	   450.,   210.,   -30.,  -270.,  -510.,  -750.,  -990.]

yOr=[-304.86625, -304.86625, -304.86625, -304.86625, -304.86625, -304.86625, -304.86625, 
	 -110.86125, -110.86125, -110.86125, -110.86125, -110.86125, -110.86125, -110.86125, 
	  110.86125,  110.86125,  110.86125,  110.86125,  110.86125,  110.86125,  110.86125,  
	  304.86625,  304.86625,  304.86625,  304.86625,  304.86625,  304.86625,  304.86625]
     
zOr=[ 44.59135295,  15.71811112,   7.91709484,	23.36712928,  55.63501811,  89.67552498, 101.83138369,	
	  15.20353847,	-0.12719265,   7.78022685,  36.79861354,  76.18874766, 106.59937932, 100.0672262 ,   
	  9.39975644,	-0.39299988,  13.32798554,  46.28252123,  85.57837979, 109.71130313,  88.56500087,  
	  35.8405432 ,	11.56006271,   9.70391668,  30.29793287,  64.75589628,	95.87956614,  97.85866817]
	
theta=[4.07607829, 2.66563388, 2.41193744, 3.28228879, 5.02179851, 7.15339301, 8.97781174, 
	   2.3441875 , 1.59544546, 2.01487151, 3.45867026, 5.56086116, 7.73327673, 9.16556328, 
	   2.04796139, 1.59259861, 2.29028171, 3.94166874, 6.12523242, 8.19725879, 9.29184753, 
	   3.62153782, 2.49122075, 2.53029745, 3.65051825, 5.54145005, 7.6704717 , 9.28277606]
       
phi=[ 145.08422394,  137.52705889,   93.45181427,   49.74101367,   25.02017254,   19.65812259,	 15.76683388,  
	  157.08333079,  121.75762408,	 70.63720544,	31.47605658,   13.77141109,    8.76364557,   10.56367995, 
	 -151.93686042, -109.06210874,	-59.04273369,  -25.06851156,  -12.07246355,   -6.73076401,  -22.53723228, 
	 -149.32189682, -128.1790711 ,	-81.45041879,  -41.45407157,  -22.251526  ,  -19.64734512,  -11.18929411]

xOr,yOr,zOr,theta,phi = map(np.array, (xOr,yOr,zOr,theta,phi))



################################################################
#################### CALCULANDO PARAMETROS #####################
################################################################


# Selecionando cornetas pelos indices
xOr = xOr[horn_idxs]
yOr = yOr[horn_idxs]
zOr = zOr[horn_idxs]
theta = theta[horn_idxs]
phi = phi[horn_idxs]

# Pegando centros dos feixes
horn_centers_datapath = "all_horn_centers.csv"
horn_centers = pd.read_csv(horn_centers_datapath)
u_centers = np.array(horn_centers["u"])[horn_idxs]
v_centers = np.array(horn_centers["v"])[horn_idxs]

# Criando lista de frequencias
Freq	     = [str(f)+" GHz" for f in Fs]
TaperAngle   = taper_angle(Fs) 
taper	     = -12.0	# dB

# Criando csv para resultados desejados
resultsFileName = jobNamePrefix+"_results.csv"
resultados	= open(resultsFileName,"tw+")
resultados.write("Job,x,y,z,theta,phi,I,u,v\n")
resultados.close()


print("Frequency list: {}".format(Freq))
index = 1
Ntotal = len(Freq) * len(Polarisation) * len(xOr) * len(Nps)

for pol in Polarisation:
	for freq,taperA in zip(Freq,TaperAngle):
		for ix,iy,iz,itheta,iphi,iu,iv in zip(xOr,yOr,zOr,theta,phi,u_centers,v_centers):
			for Npoints in Nps:
			
				u_min=iu-d_uv
				u_max=iu+d_uv
				v_min=iv-d_uv
				v_max=iv+d_uv
		
				print("\n\n\n==============================================")
				print("{}/{}) Executing pol = {}, "
					  "freq = {}".format(index,Ntotal, pol,freq))
				print("Horn: (xOr, yOr, zOr) = ({}, {}, {})\n	   "
					  "(theta, phi)    = ({}, {})".format(ix,iy,iz,itheta,iphi))
				print("Area: u=[{}, {}], v=[{}, {}], "
					  "Npoints = {}^2\n".format(u_min,u_max,v_min,v_max,Npoints))
				
				rz=R.from_euler('zyz',[iphi,itheta,-iphi],degrees=True)# calculate rotation matrix
				rzm=rz.as_matrix()*np.array([[1,-1,-1],[-1,1,1],[-1,1,1]]) #rotate Grasp coordinate system matrix
				x=rzm[0] # Extracts x axis new coordinates after rotation
				y=rzm[1] # Extracts y axis new coordinates after rotation
				dw=os.getcwd() #get local filesystem directory name
				dwO=dw +"/batch"#get working directory name
				#dwFileBase=(jobNamePrefix+"_"+str(int(ix))+"_"+str(int(iy))+ "_"+pol+ "_" + str(1000*float(freq.split()[0]))).replace("-","MINUS")
				dwFileBase=(jobNamePrefix+
							"_"+str(int(ix))+
							"_"+str(int(iy))+
							"_"+freq.replace(" ","")+
							"_"+str(Npoints)+"pts").replace("-","MINUS")
				dwD=dwFileBase
				dwFile=dwD+"//batch"
				dwFileTor=dwFile+".tor"
				dwFileGxp=dwFile+".gxp"
				shutil.copytree(dwO, dwD)#os.system("cp -r {} {}".format(dwO, dwD)) #shutil.copytree(dwO, dwD)
				tor   = open(dwFileTor,"rt")
				#open and replace TOR file
				lines = tor.read()
				tor.close()
				line0	 ="taper_angle      : XXX1,"
				line1	 ="taper            : XXX1"
				line2	 ="frequency_list   : sequence(XXX.X MHz)"
				line3	 ="origin           : struct(x: %%%% m, y: %%%% m, z: %%%% m),"
				line4	 ="x_axis           : struct(x: $$$$, y: $$$$, z: $$$$),"
				line5	 ="y_axis           : struct(x: &&&&, y: &&&&, z: &&&&),"
				line6	 ="taper_angle      : XXX2,"
				line7	 ="taper            : XXX2,"
				line8	 ="polarisation     : XXX2"
				line9	 ="x_range          : struct(start: -X.X, end: X.X, np: XXX),"
				line10	 ="y_range          : struct(start: -X.X, end: X.X, np: XXX),"
				lineN = np.array([line0,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10])
				newline0 ="taper_angle      : {},".format(taperA)
				newline1 ="taper            : {}".format(taper)
				newline2 ="frequency_list   : sequence({})".format(freq)
				newline3 ="origin           : struct(x: {} cm, y: {} cm, z: {} cm),".format(ix,iy,iz)
				newline4 ="x_axis           : struct(x: {}, y: {}, z: {}),".format(x[0],x[1],x[2])
				newline5 ="y_axis           : struct(x: {}, y: {}, z: {}),".format(y[0],y[1],y[2])
				newline6 ="taper_angle      : {},".format(taperA)
				newline7 ="taper            : {},".format(taper)
				newline8 ="polarisation     : {}".format(pol)
				newline9 ="x_range          : struct(start: {}, end: {}, np: {}),".format(u_min,u_max,Npoints)
				newline10="y_range          : struct(start: {}, end: {}, np: {}),".format(v_min,v_max,Npoints)
				newlineN = np.array([newline0,newline1,newline2,newline3,newline4,newline5,newline6,newline7,newline8,newline9,newline10])
				for line,newline in zip(lineN,newlineN):
					lines = lines.replace(line,newline)
				tor= open(dwFileTor,"wt")
				tor.write(lines)
				tor.close()
				
				os.chdir(dwFileBase)
				#calculate peek position
				os.system(realpath+dwFileBase+".out "+dwFileBase+".log")
				resultadoRaw=pd.DataFrame(data=(np.loadtxt(open("spherical_grid.grd","rt").readlines()[17:])), columns=['a', 'b', 'c','d'])
				data = np.array(np.log10(np.sqrt(np.power(resultadoRaw['a'],2)+np.power(resultadoRaw['b'],2)))*20)
				peek = data.max()
				data = data.reshape(Npoints,Npoints)
				peek_idxs = np.nonzero(data==peek)
				peek_u = u_min+(u_max-u_min)*peek_idxs[1][0]/(Npoints-1)
				peek_v = v_min+(v_max-v_min)*peek_idxs[0][0]/(Npoints-1)
				os.chdir(dw)
				#write calculated data to result file
				resultados=open(resultsFileName,"at+")
				#resultados.write(dwFileBase+","+str(ix)+","+str(iy)+","+str(iz)+","+str(itheta)+","+str(iphi)+","+str(peek)+"\n")
				resultados.write(dwFileBase+",{0},{1},{2},{3},{4},{5},{6},{7}\n".format(ix,iy,iz,itheta,iphi,peek,peek_u,peek_v))
				resultados.close()
				#return to base diretory to start a new step
				path = os.getcwd()
				pathin	= os.path.join(path,dwFileBase)
				pathout = os.path.join(path,"spherical_grid")
				#grd2txt(pathin,pathout,461,index)
				
				index+=1



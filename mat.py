# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 18:44:11 2018

@author: rkhiar
"""


import numpy as np
import pandas as pd
import decimal as Decimal

########################
#Variable definition
########################
v1=np.array([1,2])
v2=np.array([3,4])
w1=np.array([[1,-1,1,-1],[2,4,-4,-2],[3,-2,-3,2]])
w2=np.array([[1,2],[3,4],[5,6]])


#######################################
#######################################
#catch the nbrows of a matrix
def recupNbLignes(objet):
    myList=list(objet.shape)
    return myList[0]


#catch the nbcols of a matrix
def recupNbCol(objet):
    myList=list(objet.shape)
    return myList[1]
#######################################
#######################################
    

#require two vectors of the same lenght 
def addVec(vec1,vec2):
    i=0
    v=np.zeros((len(vec1)))
    while i<len(vec1):
        v[i]=vec1[i]+vec2[i]
        i+=1
    return v

#require two vectors of the same lenght 
def substractVec(vec1,vec2):
    i=0
    v=np.zeros((len(vec1)))
    while i<len(vec1):
        v[i]=vec1[i]-vec2[i]
        i+=1
    return v

#scalar vector product 
def lambdaVec(a,vec):
    i=0
    v=np.zeros((len(vec)))
    while i<len(vec):
        v[i]=vec[i]*a
        i+=1
    return v

#Matrices transpose
def transposeMatrix(mat):
    r=0
    v=np.zeros((recupNbLignes(mat),recupNbCol(mat)))
    while r<recupNbLignes(mat):
        c=0
        while c<recupNbCol(mat):
            v[r,c]=mat[c,r]
            c+=1
        r+=1
    return v

#resultat=np.array(transposeMatrix(w1), dtype='int32')
#print(resultat)

#Vectors dot product
def dotproduct(vec1,vec2):
    i=0
    dot=0
    while i<len(vec1):
       dot+=vec1[i]*vec2[i]
       i+=1
    return dot

#resultat=dotproduct(v1,v2)
#print(resultat)

#require nbrows of mat1 equals nblines of mat2
def matmul(mat1, mat2):
    v=np.zeros((recupNbLignes(mat1),recupNbCol(mat2)))
    nbr1=recupNbLignes(mat1)
    nbr2=recupNbLignes(mat2)
    nbc1=recupNbCol(mat1)
    nbc2=recupNbCol(mat2)
    
    r=0      
    while r<nbr1:
        c=0
        while c<nbc2:
            k=0
            while k<nbc1 and k<nbr2:
                v[r,c]+=mat1[r,k]*mat2[k,c]
                k+=1
            c+=1
        r+=1
    return v


#resultat=np.array(matmul(w1,w2), dtype='int32')
#print(resultat)



#Gauss methode
def Gauss(mat1):
    #v=np.zeros((recupNbLignes(mat1),recupNbCol(mat1)))
    v=np.array(mat1, dtype='float64')
    nbr1=recupNbLignes(mat1)
    nbc1=recupNbCol(mat1)
        
    r=0      
    while r<nbr1:
        c=0
        while c<nbc1:
                        
            if v[r,c]==0:
                c+=1
            else:
                k=0
                scal1=v[r,c]
                while k<nbc1:
                    v[r,k]=v[r,k]/scal1
                    k+=1
                
                f=0
                while f<nbr1:
                    if f==c:
                        f+=1
                    else:
                        scal0=-v[f,c]
                        i=0
                        while i<nbc1:
                            v[f,i]=v[f,i]+(v[r,i]*scal0)
                            i+=1
                        f+=1
                break
                            
        r+=1
    return v




#########################################################################################################################	
#########################################################################################################################	
#########################################################################################################################


##############################
#Single value decomposition
#############################
    
def svd(m):
    
    Uvals, Uvects=np.linalg.eig(m.dot(m.T))
    Vvals, Vvects=np.linalg.eig(m.T.dot(m))
    Sigma=np.array(Vvals,dtype=int)
    U=normer(np.array(Uvects,dtype=float))
    V=normer(np.array(Vvects.T,dtype=float))

    return U, np.sqrt(Sigma), V
   
    

#methode permettant de normer un les vecteurs d'une matrice
    
def normer(m):
    
    #decoupage de la matrice en vecteurs
    maListe=list(np.hsplit(m,recupNbCol(m)))
    
    #normage des vecteurs et reconstruction de la matrice normée
    i=0
    mat=np.zeros((recupNbLignes(m),1))
        
    while i<len(maListe):
        maListe[i]=maListe[i]/np.linalg.norm(maListe[i])
        mat=np.hstack((mat,maListe[i]))
        i+=1
   
    #res contien la matrice normée     
    res=np.array(mat[:,1:])
           
    return(res)
             
  

#matTest=np.array([[3,2,2],[2,3,-2]])

#print(svd(matTest))

#print(np.linalg.svd(matTest))

#print(np.linalg.eig(matTest.T.dot(matTest)))




########################################"
## APPLICATION SVD
############################################

var=pd.read_csv("rattingsTest.csv", sep=",")
a=np.array(var, dtype="int")

#print(a)

####################################
#Construction de l'entete users  
####################################

i=0
b=np.zeros((1,1))
#print(b)

while i<recupNbLignes(a):
    TestPres=False
    j=0
    while j<recupNbLignes(b):
        
        if a[i,0]==b[j,0]:
            TestPres=True
            break
        j+=1
    
    if TestPres==False:
        b=np.vstack((b,a[i,0]))
        
    i+=1
        
#print(b)
    


####################################
#Construction de l'entete films    
####################################
   
g=0
c=np.zeros((recupNbLignes(b),1))

while g<recupNbLignes(a):
    TestPresFilm=False
    h=0
    while h<recupNbCol(b):
        
        if a[g,1]==b[0,h]:
            TestPresFilm=True
            break
            
        h+=1
    
    if TestPresFilm==False:
        c=np.zeros((recupNbLignes(b),1))
        c[0,0]=a[g,1]
        b=np.hstack((b, c))
        
    g+=1
        


  
################################3
#Remplissage de la matrice
####################################

i=0

while i<recupNbLignes(a):
    j=0
    while j<recupNbLignes(b):
        if a[i,0]==b[j,0]:
            k=0
            while k<recupNbCol(b):
                if a[i,1]==b[0,k]:
                    b[j,k]=a[i,2]
                k+=1
        j+=1
    i+=1
    
#print(b)
    

print("""
      
      """)

#print(b[1:,1:])

#print(np.linalg.svd(b[1:,1:]))




#########################################################################################################################	
#########################################################################################################################	
#########################################################################################################################




# Données input (x) et output(y)
#x=np.array([[1], [1.5], [2], [2.5], [3.5], [4], [4.5], [5], [5.5], [6]])
#y=np.array([[1.10], [1.15], [1.20], [1.195], [1.30], [1.40], [1.38], [1.50], [1.47], [1.60]])

#X=np.hstack((x,np.ones((recupNbLignes(x),1))))


#w=np.array([[1],[1]], dtype='float64')



# fonction mse
def mse(inputs, outputs, weights):
    
    return 1/recupNbLignes(inputs)*np.sum(np.square(outputs-(weights[0,0]*inputs[:,0].reshape(recupNbLignes(inputs),1)+weights[1,0])))


# Fonction Gradient
def gradient(inputs, outputs, weights):
    #derivé de la fonction MSE par rapport a w1
    DmseW1=2/recupNbLignes(inputs)*np.sum((-inputs[:,0].reshape(recupNbLignes(inputs),1)*outputs)+(inputs[:,0].reshape(recupNbLignes(inputs),1)*weights[1,0])+(weights[0,0]*np.square(inputs[:,0].reshape(recupNbLignes(inputs),1))))

    #derivé de la fonction MSE par rapport a b
    DmseB=2/recupNbLignes(inputs)*np.sum((-outputs)+(inputs[:,0].reshape(recupNbLignes(inputs),1)*weights[0,0])+(weights[1,0]))
    
    # Definition du gradient
    grad=np.array([[round(DmseW1,3)],[round(DmseB,3)]])
    
    return(grad)
    
    

def gradientDescent(inputs, outputs, weights):
    i=2000
     
    while i>0:
        
        weights-=0.001*gradient(inputs, outputs, weights)
        i-=1
    
    return(weights, round(mse(inputs, outputs, weights),3))
    
    
#print(gradientDescent(X,y))
#print(""" Mon gradient """)
#print(gradient(X, y, w))

#print(""" Mon gradient descent """)
#print(gradientDescent(X, y, w))


#print(""" MSE """)
#print(round(mse(X,y,w),3))
    


###########################################################################################################
###########################################################################################################
###  Derivation de la MSE pour un model multidim
###########################################################################################################
###########################################################################################################




# Données input (x) et output(y)
x1=np.array([[1], [1.5], [2], [2.5], [3.5], [4], [4.5], [5], [5.5], [6]])
x2=np.array([[2], [2.5], [3], [3.5], [4.5], [5], [5.5], [6], [6.5], [7]])

y=np.array([[1.10], [1.15], [1.20], [1.195], [1.30], [1.40], [1.38], [1.50], [1.47], [1.60]])

zOne=np.ones((recupNbLignes(x1),1))

X=np.hstack((x1, x2, zOne))

VectW=np.ones((recupNbCol(X),1))

'''VectW[0,0]=0.1
VectW[1,0]=0.98

print(VectW)'''


#print(X[:,-1])

# fonction mse
def MultiMse(inputs, outputs, weights):
    
    mse = 1/recupNbLignes(inputs)*np.sum(np.square(outputs-(inputs[:,:-1].dot(weights[:-1,:])+weights[-1,0])))
    
    return round(mse,3)

def DerivB (inputs, outputs, weights): 
    
    DmseB=2/recupNbLignes(inputs)*np.sum((-outputs)+(inputs[:,:-1].dot(weights[:-1,:]))+(weights[-1,0]))      
    
    return round(DmseB,3)


def DerivW (inputs, outputs, weights): 
    
    GradW=np.zeros((recupNbCol(inputs)-1,1))
    
    i=0
    while i<recupNbCol(inputs)-1:
        
        xi=inputs[:,i].reshape(recupNbLignes(inputs),1)
       
        xTrt=np.hstack(((inputs[:,0:i]),inputs[:,i+1:-1]))
        wTrt=np.vstack(((VectW[0:i,:]),VectW[i+1:-1,:]))
        
                            
        GradW[i] = round(2/recupNbLignes(inputs)*np.sum(-xi*outputs + xi*(xTrt.dot(wTrt)) +xi*weights[-1,0]+weights[i,0]*np.square(xi)), 3)
               
        i+=1
        
    return GradW


# Fonction Gradient
def MultiGradient(inputs, outputs, weights):
    #derivé de la fonction MSE par rapport a w1
    DmseW1=DerivW(inputs, outputs, weights)

    #derivé de la fonction MSE par rapport a b
    DmseB=DerivB(inputs, outputs, weights)
    
    # Definition du gradient
    grad=np.vstack((DmseW1,DmseB))
    
    print(""" mon grad""")
    print(grad)
    
    return(grad)




def MultiGradientDescent(inputs, outputs, weights):
    i=2000
     
    while i>0:
        
        weights-=0.001*MultiGradient(inputs, outputs, weights)
        i-=1
    
    return(weights, MultiMse(inputs, outputs, weights))


#print(MultiGradientDescent(X, y, VectW))


#print(""" Mon MultiGradientDescent """)
#print(MultiGradientDescent(X, y, VectW))




################################################################################"
####  Classification Multiple
#################################################################################

# Definition des entrées (x1--> poid  / x2---> Taille)
x1=np.array([[40, 45, 50, 55, 60, 65, 70, 75, 80]]).reshape((9,1))
x2=np.array([[145, 150, 155, 160, 165, 170, 175, 180, 190]]).reshape((9,1))

zOne=np.ones((recupNbLignes(x1),1))

# Definition de la matrice Y contenant le veritable genre de chaque point
P=np.array([[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0]])

#construction de la matrice X
X=np.hstack((x1, x2, zOne))

# Definition de la matrice de poids
w=np.array([[1,0.5],[1,0.5],[1,1]])

SomExp=np.ones((2,1))


#dEfinition de ma propre fonction d'arrondi !!!! le round() c'est de la merde *
def myRound(s): 
    i=0
    d=np.ones((recupNbLignes(s),recupNbCol(s)-1))
    while i<recupNbCol(s):
        d=np.hstack((d,np.array([float(str(elt)[0:5].replace('[','')) for elt in s[:,i]]).reshape((recupNbLignes(s),recupNbCol(s)-1))))
        i+=1
    return d[:,1:]


#Definition de la fonction Score
def score(weights, inputs):
    return inputs.dot(weights)
 

#Definition de la fonction softmax
def softmax(sc):
          
    matSoft=np.exp(sc-np.max(sc,axis=1).reshape((recupNbLignes(sc),1)))/np.sum(np.exp(sc-np.max(sc,axis=1).reshape((recupNbLignes(sc),1))), axis=1).reshape((recupNbLignes(sc),1))
    
    return matSoft


#Definition de la fonction de perte 
def crossEntropy(outputs, inputs, weights):
    
    res=(1/recupNbLignes(outputs))*(np.sum(outputs*np.log(softmax(score(weights,inputs)))))
    
    return res



# Dervié de la fonction cross entropy
def derivCross(weights, soft, inputs, outputs):
    derive=np.ones((recupNbLignes(soft), 1))
    res=np.array(weights)
    
    j=0
    while j<recupNbCol(weights):
        i=0
        while i<recupNbLignes(weights):
            derive=np.ones((recupNbLignes(soft), 1))   
            
            if j>0:
                            
                derive=np.hstack((derive, soft[:,:j]*(-1*soft[:,j]*inputs[:,i]).reshape((recupNbLignes(soft), 1))))
           
            derive=np.hstack((derive, (soft[:,j]*(1-soft[:,j])*inputs[:,i]).reshape(((recupNbLignes(soft), 1)))))
                
            if j<recupNbCol(soft)-1:
                derive=np.hstack((derive, (soft[:,j+1:])*(-1*soft[:,j]*inputs[:,i]).reshape((recupNbLignes(soft), 1))))
          
            #res.append((1/recupNbLignes(inputs))*np.sum(  (-outputs/soft)*(derive[:,1:])  ))
            res[i,j]=(1/recupNbLignes(inputs))*np.sum(  (-outputs/soft)*(derive[:,1:])  )
            
            i+=1
        j+=1
        
    return res




def softGradientDescent(weights, soft, inputs, outputs):
    i=2000
    while i>0:
        
        weights-=0.001*derivCross(weights, softmax(score(weights,inputs)) ,inputs, outputs)
        i-=1
    
    return(weights, crossEntropy(outputs, inputs, weights))



#print("""ma derivée """)
#print(derivCross(w, softmax(score(w,X)) ,X, P))


#print("""gradient descent""")
#print(softGradientDescent(w, softmax(score(w,X)), X, P))

w2=np.array([[1.3579545 , 0.1420455 ],
       [0.5349946 , 0.9650054 ],
       [0.99199451, 1.00800549]])

print(np.sum(softmax(score(w2,X)), axis=1))



#print("""score""")
#print(score(w,X))

#print(""" Exp score""")
#print(myRound(np.exp(score(w,X))))

#print("""Some Exp score""")
#print(np.sum(np.exp(score(w,X)), axis=1).reshape((recupNbLignes(score(w,X)),1)))

#print(""" Softmax """)
#print(softmax(score(w,X)))

#print(""" ma cross entropy """)
#print(crossEntropy(P,X,w))



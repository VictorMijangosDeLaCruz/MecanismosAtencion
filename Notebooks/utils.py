import numpy as np

def example_data(k, r0=1, r1=3):
    X1 = [np.array([r0*np.cos(t),r0*np.sin(t)]) for t in range(0,k)]
    X2 = [np.array([r1*np.cos(t),r1*np.sin(t)]) for t in range(0,k)]
    X = np.concatenate((X1,X2))
    n,d = X.shape
    Y = np.zeros(2*k)
    Y[k:] += 1
    noise = np.array([np.random.normal(0,1,2) for i in range(n)])
    X += 0.5*noise 

    #Seprara en train y en test
    #x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)
    
    return X, Y

    #Visualización de train set
    #plt.scatter(x_train[:,0], x_train[:,1],c=y_train,s=1)
    #plt.show()
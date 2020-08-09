import numpy as np
from scipy import special
import scipy.linalg as linalg
def PyMastic(q,a,x,z,H,E,nu, ZRO=1e-3, isBounded = [1,1], iteration = 25, inverser = 'solve'):
    '''
    PyMastic calculates the respnse of a multi-layered elastic system subjected to a circular load. 
    
    Parameters
    ----------
    q : float or int.
        tire pressure level
    a : float or int.
        radius of tire
    x : numpy array or list (1D)
        horizontal points to be analyzed
    z : numpy array or list (1D)
        vertical points to be analyzed
    H : numpy array or list (1D)
        thickness of each layer
    E : numpy array or list (1D)
        modulus of each layer
    nu : numpy array or list (1D)
        Poisson's Ratio of each layer
    isBounded = list,
        condition of each interface. bounded=1  not bounded=0
    iteration : int
        number of iteration before convergence. The default is 25
    inverser : string, optional
        the solver for the inverse matrix. The default is 'solve'

    Returns
    -------
    Response : Disctionary:
                {"Displacement_Z": displacementW, 
               "Displacement_H": displacementU,
               "Stress_Z": sigmaZ,
               "Stress_R": sigmaR,
               "Stress_T": sigmaT,
               "Strain_Z": epsZ,
               "Strain_R": epsR,
               "Strain_T": epsT
               }
        The calculated pavement response for strains, stresses, and displacements x, and z.
        Tn the final dictionary the columns are x offsets and the rows are z depth.
        
    @ Author: Mostafa Nakhaei    
   
    @ Copyright:
    Copyright (c) 2020, Mostafa Nakhaei
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
    
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution
      
    * Neither the name of  nor the names of its
      contributors may be used to endorse or promote products derived from this
      software without specific prior written permission.
      
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    '''
    x = np.array(x, dtype=np.float64)
    x[x == 0] = 1e-6
    z = np.array(z, dtype=np.float64)
    z[z == 0] = 1e-6
    H = np.array(H, dtype=np.float64)
    E = np.array(E, dtype=np.float64) * 1000
    nu = np.array(nu, dtype=np.float64)
    
    firstKindFirstOrder = np.array([3.83170597020751,7.01558666981562,10.1734681350627,13.3236919363142,16.4706300508776,19.6158585104682,22.7600843805928, 25.9036720876184,29.0468285349169,32.1896799109744,35.3323075500839,38.4747662347716,41.6170942128145,44.759318997652, 47.9014608871855,51.0435351835715,54.1855536410613,57.3275254379010,60.4694578453475,63.6113566984812,66.7532267340985, 69.8950718374958,73.0368952255738,76.1786995846415,79.3204871754763,82.4622599143736,85.6040194363502,88.7457671449263, 91.8875042516950,95.0292318080447,98.1709507307908,101.312661823039,104.454365791283,107.596063259509,110.737754780899,113.879440847595,117.021121898892,120.162798328149,123.304470488636,126.446138698517,129.587803245104,132.729464388510,135.871122364789,139.012777388660,142.154429655859,145.296079345196,148.437726620342,151.579371631401,154.721014516286,157.862655401930,161.004294405362,164.145931634650,167.287567189744,170.429201163227,173.570833640976,176.712464702764,179.854094422788,182.995722870153,186.137350109296,189.278976200376,192.420601199626,195.562225159663,198.703848129777,201.845470156191,204.987091282292,208.128711548850,211.270330994208,214.411949654462,217.553567563624,220.695184753769,223.836801255172,226.978417096429,230.120032304579,233.261646905201,236.403260922514,239.544874379470,242.686487297829,245.828099698240,248.969711600310,252.111323022669,255.252933983028,258.394544498240,261.536154584344,264.677764256622,267.819373529635,270.960982417271,274.102590932781,277.244199088815,280.385806897456,283.527414370251,286.669021518243,289.810628351994,292.952234881614,296.093841116782,299.235447066774,302.377052740478,305.518658146416,308.660263292764, 311.801868187371,314.943472837767])
    firstKindZeroOrder = np.array([2.40482555769577,5.52007811028631,8.65372791291101,11.7915344390143,14.9309177084878,18.0710639679109, 21.2116366298793,24.3524715307493,27.4934791320403,30.6346064684320,33.7758202135736,36.9170983536640,40.0584257646282, 43.1997917131767,46.3411883716618,49.4826098973978,52.6240518411150,55.7655107550200,58.9069839260809,62.0484691902272, 65.1899648002069,68.3314693298568,71.4729816035937,74.6145006437018,77.7560256303881,80.8975558711376,84.0390907769382, 87.1806298436412,90.3221726372105,93.4637187819448,96.6052679509963,99.7468198586806,102.888374254195,106.029930916452, 109.171489649805,112.313050280495,115.454612653667,118.596176630873,121.737742087951,124.879308913233,128.020877006008, 131.162446275214,134.304016638305,137.445588020284,140.587160352854,143.728733573690,146.870307625797,150.011882456955, 153.153458019228,156.295034268534,159.436611164263,162.578188668947,165.719766747955,168.861345369236,172.002924503078, 175.144504121903,178.286084200074,181.427664713731,184.569245640639,187.710826960049,190.852408652582,193.993990700109, 197.135573085661,200.277155793332,203.418738808199,206.560322116244,209.701905704294,212.843489559950,215.985073671534, 219.126658028041,222.268242619084,225.409827434859,228.551412466099,231.692997704039,234.834583140383,237.976168767276, 241.117754577268,244.259340563296,247.400926718653,250.542513036970,253.684099512193,256.825686138564,259.967272910605, 263.108859823096,266.250446871066,269.392034049776,272.533621354705,275.675208781537,278.816796326153,281.958383984615, 285.099971753160,288.241559628188,291.383147606255,294.524735684065,297.666323858459,300.807912126411,303.949500485021, 307.091088931505,310.232677463195,313.374266077528])
    nLayers = len(nu)
    sumH = sum(H)
    Lamda = np.hstack((0, np.cumsum(H)/sumH, 1e3))
    L = z/sumH
    ro = x/sumH
    alpha = a/sumH
    ind = np.zeros(z.shape)
    
    firstKindZeroOrder = firstKindZeroOrder / ro[:, None]
    firstKindFirstOrder = firstKindFirstOrder / alpha
    firstKindZeroOrder = firstKindZeroOrder.T
    BesselZeros = np.hstack((np.array([0]), firstKindZeroOrder.flatten(), firstKindFirstOrder.flatten())).flatten()
    BesselZeros = np.sort(BesselZeros)
    D1 = (BesselZeros[1]-BesselZeros[0]) / 6 - 0.00001
    D2 = (BesselZeros[2]-BesselZeros[1]) / 2 - 0.00001
    AUX1 = np.arange(BesselZeros[0], BesselZeros[1], D1)
    AUX2 = np.arange(BesselZeros[1], BesselZeros[2], D2)
    mValues = np.hstack((AUX1, AUX2[1:], BesselZeros[3:iteration])).flatten()
    getDiff = np.diff(mValues)
    mValuesMatrix = np.vstack((mValues, mValues, mValues, mValues)).T
    ftGauss = np.zeros((4, mValuesMatrix.shape[0]-1))
    coefficient = np.zeros((mValuesMatrix.shape[0]-1, 4))
    coefficient[:,0] =  getDiff / 2 - 0.86114 * (getDiff / 2)
    coefficient[:,1] =  getDiff / 2 - 0.33998 * (getDiff / 2)
    coefficient[:,2] =  getDiff / 2 + 0.33998 * (getDiff / 2)
    coefficient[:,3] =  getDiff / 2 + 0.86114 * (getDiff / 2)
    ftGauss[0, :] =   0.34786 * (getDiff / 2);
    ftGauss[1, :] =   0.65215 * (getDiff / 2);
    ftGauss[2, :] =   0.65215 * (getDiff / 2);
    ftGauss[3, :]=   0.34786 * (getDiff / 2);
    ft = ftGauss.flatten(order="F")
    mFinal =  mValuesMatrix[0:-1,:] + coefficient
    mNotSorted = mFinal.flatten(order="F")
    m = np.sort(mNotSorted)
    # m = np.array([1e-10, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60,
    #               1.80, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80,
    #               4.00, 4.20, 4.40, 4.60, 4.80, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50,
    #               8.00, 8.50, 9.00, 9.50, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00,
    #               16.00, 17.00, 18.00, 19.00, 20.00, 25.00, 30.00, 35.00, 40.00,
    #               45.00, 50.00, 55.00, 60.00, 65.00, 70.00, 75.00, 80.00, 85.00,
    #               90.00, 95.00, 100.00, 110.00, 120.00, 130.00, 140.00, 150.00,
    #               160.00, 170.00, 180.00, 190.00, 200.00, 210.00, 220.00, 230.00,
    #               240.00, 250.00, 260.00, 270.00, 280.00, 290.00, 300.00, 350.00,
    #               400.00, 450.00, 500.00, 600.00, 700.00, 800.00, 900.00, 1000.00,
    #               2000.00, 10000.00, 100000.00])

    for i in range(len(ind)):
        ind[i] = np.where(Lamda > L[i])[0][0]

    A = np.zeros((len(m),nLayers))
    B = np.zeros((len(m),nLayers))
    C = np.zeros((len(m),nLayers))
    D = np.zeros((len(m),nLayers))
    
    ########################################################################################################
    ################################### BOUNDRAY CONDITION: START ##########################################
    ########################################################################################################
    A_BC = np.zeros((nLayers, 1))
    B_BC = np.zeros((nLayers, 1))
    C_BC = np.zeros((nLayers, 1))
    D_BC = np.zeros((nLayers, 1))
    LeftMatrix= np.zeros((nLayers-1, 4, 4))
    RightMatrix = np.zeros((nLayers-1, 4, 4))
    InvLeftMatrix= np.zeros((nLayers-1, 4, 4))
    solved_matrix = np.zeros((nLayers-1, 4, 4))
    H_BC = np.hstack((H, max(H)*1e3))
    Lamda_BC = np.cumsum(H_BC)/sumH        
    R = E[0:-1] / E[1:] *((1+nu[1:]) / (1 + nu[0:-1]))
    
    for j in range(len(m)):
        
        LeftMatrix1 = np.array([[np.exp(-m[j] * Lamda_BC[0]), 1], [np.exp(-m[j] * Lamda_BC[0]), -1]])
        RightMatrix1 = np.array([[-(1 - 2 * nu[0]) * np.exp(-m[j] * Lamda_BC[0]), 1 - 2 * nu[0]], [2 * nu[0] * np.exp(-m[j] * Lamda_BC[0]), 2 * nu[0]]])
        dLambda = np.diff(np.hstack((0, Lamda_BC)))
        F = np.exp(-m[j] * dLambda)
        for i in range(nLayers-1):
            if int(isBounded[i]) == 1:
                LeftMatrix[i,:,:] = np.array([[1, F[i], -(1 - 2 * nu[i] - m[j] * Lamda_BC[i]), (1 - 2 * nu[i] + m[j] * Lamda_BC[i]) * F[i]],
                                       [1, -F[i], 2 * nu[i] + m[j] * Lamda_BC[i], (2 * nu[i]- m[j] * Lamda_BC[i]) * F[i]],
                                       [1, F[i], 1 + m[j] * Lamda_BC[i], -(1 - m[j] * Lamda_BC[i]) * F[i]],
                                       [1, -F[i], -(2 - 4 * nu[i] - m[j] * Lamda_BC[i]), -(2 - 4 * nu[i] + m[j] * Lamda_BC[i]) * F[i]]], dtype=np.float64)

                if inverser == 'inv':
                    InvLeftMatrix[i,:,:] = np.linalg.inv(LeftMatrix[i])
                elif inverser == 'pinv':
                    InvLeftMatrix[i,:,:] = np.linalg.pinv(LeftMatrix[i])
                elif inverser == 'lu':
                    LU = linalg.lu_factor(LeftMatrix[i]) 
                elif inverser == 'svd':
                    U,s,V = np.linalg.svd(LeftMatrix[i])
                    InvLeftMatrix[i,:,:] = np.transpose(V) @ np.diag(1/s) @ np.transpose(U)
                    
                RightMatrix[i,:,:] = np.array([[F[i+1], 1, -(1 - 2 * nu[i+1] - m[j] * Lamda_BC[i]) * F[i+1], 1 - 2 * nu[i+1] + m[j] * Lamda_BC[i]],
                                        [F[i+1], -1, (2 * nu[i+1] + m[j] * Lamda_BC[i]) * F[i+1], 2 * nu[i+1] - m[j] * Lamda_BC[i]],
                                        [R[i] * F[i+1], R[i], (1 + m[j] * Lamda_BC[i]) * R[i] * F[i+1], -(1 - m[j] * Lamda_BC[i]) * R[i]],
                                        [R[i] * F[i+1], -R[i], -(2 - 4 * nu[i+1] - m[j] * Lamda_BC[i]) * R[i] * F[i+1],-(2 - 4 * nu[i+1] + m[j] * Lamda_BC[i]) * R[i]]])
                if  inverser == 'pinv' or  inverser == 'inv':
                    solved_matrix[i,:,:] = np.dot(InvLeftMatrix[i,:,:], RightMatrix[i,:,:])
                elif inverser == 'svd':
                    solved_matrix[i,:,:] = InvLeftMatrix[i,:,:] @ RightMatrix[i,:,:]
                elif inverser == 'lu':
                    solved_matrix[i,:,:] = linalg.lu_solve(LU, RightMatrix[i,:,:])
                    
                elif inverser == 'solve':
                    solved_matrix[i,:,:] = np.linalg.solve(LeftMatrix[i], RightMatrix[i,:,:])
            ##--------------------------------------------------- frictionless: START ---------------------------------------------------## 
            elif int(isBounded[i]) == 0:
                LeftMatrix[i,:,:] = np.array([[1, F[i], -(1 - 2 * nu[i] - m[j] * Lamda_BC[i]), (1 - 2 * nu[i] + m[j] * Lamda_BC[i]) *F[i]],
                                        [1, -F[i], -(2 - 4 * nu[i] - m[j] * Lamda_BC[i]), -(2 - 4 * nu[i] + m[j] * Lamda_BC[i]) * F[i]],
                                        [1, -F[i], 2 * nu[i] + m[j] * Lamda_BC[i], (2 * nu[i]- m[j] * Lamda_BC[i]) * F[i]],
                                        [ZRO, ZRO, ZRO, ZRO]], dtype=np.float64)

                if inverser == 'inv':
                    InvLeftMatrix[i,:,:] = np.linalg.inv(LeftMatrix[i])
                elif inverser == 'pinv':
                    InvLeftMatrix[i,:,:] = np.linalg.pinv(LeftMatrix[i])
                elif inverser == 'lu':
                    LU = linalg.lu_factor(LeftMatrix[i]) 
                elif inverser == 'svd':
                    U,s,V = np.linalg.svd(LeftMatrix[i])
                    InvLeftMatrix[i,:,:] = np.transpose(V) @ np.diag(1/s) @ np.transpose(U)           
                # else:
                #     raise 'the solver choice is invalid, select among: np.linalg.solve  np.linalg.inv  np.linalg.pinv'
                RightMatrix[i,:,:] = np.array([[F[i+1], 1, -(1 - 2 * nu[i+1] - m[j] * Lamda_BC[i]) * F[i+1], 1 - 2 * nu[i+1] + m[j] * Lamda_BC[i]],
                                        [R[i] * F[i+1], -R[i], -(2 - 4 * nu[i+1] - m[j] * Lamda_BC[i]) * R[i] * F[i+1],-(2 - 4 * nu[i+1] + m[j] * Lamda_BC[i]) * R[i]],
                                        [ZRO, ZRO, ZRO, ZRO],
                                        [F[i+1], -1, (2 * nu[i+1] + m[j] * Lamda_BC[i]) * F[i+1], 2 * nu[i+1] - m[j] * Lamda_BC[i]]])
               
                if  inverser == 'pinv' or  inverser == 'inv' or inverser == 'svd':
                    solved_matrix[i,:,:] = np.dot(InvLeftMatrix[i,:,:], RightMatrix[i,:,:])
                elif inverser == 'lu':
                    solved_matrix[i,:,:] = linalg.lu_solve(LU, RightMatrix[i,:,:])              
                    
                elif inverser == 'solve':
                    solved_matrix[i,:,:] = np.linalg.solve(LeftMatrix[i], RightMatrix[i,:,:])     
            ##--------------------------------------------------- frictionless: END ---------------------------------------------------## 
            else:
                raise 'The boundry condition is invalid'
        
        ##--------------------------------------------------- Finidng Bn and Dn: START ---------------------------------------------------##
        BnDn_Matrix = solved_matrix[0,:,:]
        for i in range(1, nLayers-1):
             BnDn_Matrix = np.dot(BnDn_Matrix, solved_matrix[i,:,:])
        BnDn_Matrix = BnDn_Matrix[:,[1,3]] #equation B.15
        
        ########### METHOD 1: START        
        try:
            NN = np.dot(np.hstack([LeftMatrix1, RightMatrix1]), BnDn_Matrix)
            BnDn = np.linalg.solve(NN, np.array([[1],[0]]))
        except:
            MM = np.dot(np.hstack([LeftMatrix1, RightMatrix1]), BnDn_Matrix)
            NN = np.linalg.pinv(MM)
            BnDn = np.dot(NN, np.array([[1],[0]]))
            print("singular matrix, PINV was used instead, REDUCE the number of iterations")
        B_BC[-1] = BnDn[0]
        D_BC[-1] = BnDn[1]
        ########### METHOD 1: END   
        
        ########### METHOD 2: START
        MC = np.zeros((4, 1)) # based on equation B.9, method 1 is faster
        M1=np.array([[np.exp(-m[j]*Lamda_BC[0]), 1], [np.exp(-m[j]*Lamda_BC[0]), -1]])
        M2=np.array([[-(1-2*nu[0])*np.exp(-m[j]*Lamda_BC[0]), 1-2*nu[0]], [2*nu[0]*np.exp(-m[j]*Lamda_BC[0]), 2*nu[0]]])
        MC[0, 0]=M1[0, 0]*BnDn_Matrix[0, 0]+M1[0, 1]*BnDn_Matrix[1, 0]+M2[0, 0]*BnDn_Matrix[2, 0]+M2[0, 1]*BnDn_Matrix[3, 0]
        MC[1, 0]=M1[0, 0]*BnDn_Matrix[0, 1]+M1[0, 1]*BnDn_Matrix[1, 1]+M2[0, 0]*BnDn_Matrix[2, 1]+M2[0, 1]*BnDn_Matrix[3, 1]
        MC[2, 0]=M1[1, 0]*BnDn_Matrix[0, 0]+M1[1, 1]*BnDn_Matrix[1, 0]+M2[1, 0]*BnDn_Matrix[2, 0]+M2[1, 1]*BnDn_Matrix[3, 0]
        MC[3, 0]=M1[1, 0]*BnDn_Matrix[0, 1]+M1[1, 1]*BnDn_Matrix[1, 1]+M2[1, 0]*BnDn_Matrix[2, 1]+M2[1, 1]*BnDn_Matrix[3, 1]
        B_BC[-1]=MC[3, 0]/(MC[0, 0]*MC[3, 0]-MC[1, 0]*MC[2, 0])
        D_BC[-1]=1/(MC[1, 0]-MC[3, 0]*MC[0, 0]/MC[2, 0])
        ########## METHOD 2: END
        ##--------------------------------------------------- Finidng Bn and Dn: END ---------------------------------------------------## 
        
        for i in reversed(range(nLayers-1)):
            BC = np.dot(solved_matrix[i,:,:], np.vstack((A_BC[i+1], B_BC[i+1], C_BC[i+1], D_BC[i+1])))
            A_BC[i] = BC[0]
            B_BC[i] = BC[1]
            C_BC[i] = BC[2]
            D_BC[i] = BC[3]

        A[j,:] = A_BC.flatten()
        B[j,:] = B_BC.flatten()
        C[j,:] = C_BC.flatten()
        D[j,:] = D_BC.flatten()    
    ########################################################################################################
    ################################### BOUNDRAY CONDITION: START ##########################################
    ########################################################################################################
        
    sigmaR = np.zeros((len(z),len(x)), dtype=np.float64)
    sigmaT = np.zeros((len(z),len(x)), dtype=np.float64)
    sigmaZ = np.zeros((len(z),len(x)), dtype=np.float64)
    epsR = np.zeros((len(z),len(x)), dtype=np.float64)
    epsT = np.zeros((len(z),len(x)), dtype=np.float64)
    epsZ = np.zeros((len(z),len(x)), dtype=np.float64)
    displacementZ = np.zeros((len(z),len(x)), dtype=np.float64)
    displacementH = np.zeros((len(z),len(x)), dtype=np.float64)
    for j in range(len(x)):
        for i in range (len(z)):
            Rs = -1.0 * ((1.0 + nu[int(ind[i] - 1)]) / E[int(ind[i] - 1)]) *    \
                special.jv(0, m * ro[j]) *                         \
                    ((A[:, int(ind[i]-1)] - C[:, int(ind[i] - 1)] * (2 - 4 * nu[int(ind[i]-1)] - m * L[i]))          
                     * np.exp(-1 * m * (Lamda[int(ind[i])] - L[i])) - (B[:, int(ind[i]-1)] + D[:, int(ind[i]-1)] * (2 - 4 * nu[int(ind[i]-1)] + m * L[i])) *  \
                         np.exp(-1 * m * (L[i]-Lamda[int(ind[i]-1)])))
            displacementZ[i, j] = sumH * q * alpha * sum(ft * Rs * special.jv(1, m * alpha) * (1 / m))
    
    for j in range(len(x)): #TODO: check it
        for i in range (len(z)):
            Rs = (1.0 + nu[int(ind[i] - 1)]) / E[int(ind[i] - 1)] *    \
                special.jv(1, m * ro[j]) *                         \
                    ((A[:, int(ind[i]-1)] + C[:, int(ind[i] - 1)] * (1 + m * L[i]))          
                     * np.exp(-1 * m * (Lamda[int(ind[i])] - L[i])) + (B[:, int(ind[i]-1)] - D[:, int(ind[i]-1)] * (1 - m * L[i])) *  \
                         np.exp(-1 * m * (L[i]-Lamda[int(ind[i]-1)])))
            displacementH[i, j] = sumH * q * alpha * sum(ft * Rs * special.jv(1, m * alpha) * (1 / m))   
    
    
    for j in range(len(x)):
        for i in range (len(z)):
            Rs = -m * special.jv(0, m * ro[j]) * ((A[:, int(ind[i]-1)]-C[:, int(ind[i] - 1)] * (1-2*nu[int(ind[i]-1)]-m*L[i]))* \
                                                 np.exp(-m*(Lamda[int(ind[i])] - L[i])) + (B[:, int(ind[i]-1)]+D[:, int(ind[i]-1)]*(1-2*nu[int(ind[i]-1)]+m*L[i]))*\
                                                     np.exp(-m*(L[i]-Lamda[int(ind[i]-1)])))
            sigmaZ[i, j] =   -1 * q * alpha * sum(ft * Rs * special.jv(1, m * alpha) * (1 / m))
    
    for j in range(len(x)):
        for i in range (len(z)):
            Rs = (m * special.jv(0, m*ro[j]) - (1 / ro[j]) * special.jv(1, m*ro[j])) * \
                ((A[:, int(ind[i]-1)] + C[:, int(ind[i]-1)] * (1 + m * L[i])) * \
                 np.exp(-m*(Lamda[int(ind[i])] - L[i])) + (B[:, int(ind[i]-1)] - D[:,int(ind[i]-1)]*(1-m*L[i])) * \
                     np.exp(-m*(L[i]-Lamda[int(ind[i]-1)]))) + 2 * nu[int(ind[i]-1)] * m * special.jv(0, m*ro[j]) * \
                         (C[:, int(ind[i]-1)] * np.exp(-m*(Lamda[int(ind[i])] -L[i])) - D[:, int(ind[i]-1)] * np.exp(-m*(L[i]-Lamda[int(ind[i]-1)]))) 
            sigmaR[i, j] =   -1 * q * alpha * sum(ft * Rs * special.jv(1, m * alpha) * (1 / m))
        
    for j in range(len(x)):
        for i in range (len(z)):
            Rs = (1 / ro[j]) * special.jv(1,m*ro[j]) * \
                ((A[:, int(ind[i]-1)] + C[:, int(ind[i]-1)] * (1+m*L[i])) * \
                 np.exp(-m*(Lamda[int(ind[i])] -L[i])) + (B[:, int(ind[i]-1)]-D[:,int(ind[i]-1)]*(1-m*L[i])) * \
                     np.exp(-m*(L[i]-Lamda[int(ind[i]-1)]))) + 2*nu[int(ind[i]-1)]*m*special.jv(0,m*ro[j])* \
                    (C[:, int(ind[i]-1)]*np.exp(-m*(Lamda[int(ind[i])]-L[i]))-D[:, int(ind[i]-1)]* \
                     np.exp(-m*(L[i]-Lamda[int(ind[i]-1)])))
            sigmaT[i, j] =   -1 * q * alpha * sum(ft * Rs * special.jv(1, m * alpha) * (1 / m))
            
    for j in range(len(x)):
        for i in range (len(z)):
            epsZ[i,j] = 1 / E[int(ind[i]-1)] * (sigmaZ[i,j] - nu[int(ind[i]-1)] * (sigmaT[i,j] + sigmaR[i,j]))
            
    for j in range(len(x)):
        for i in range (len(z)):
            epsR[i,j] = 1 / E[int(ind[i]-1)] * (sigmaR[i,j] - nu[int(ind[i]-1)] * (sigmaZ[i,j] + sigmaT[i,j]))
   
    for j in range(len(x)):
        for i in range (len(z)):
            epsT[i,j] = 1 / E[int(ind[i]-1)] * (sigmaT[i,j] - nu[int(ind[i]-1)] * (sigmaZ[i,j] + sigmaR[i,j]))  

    Response ={"Displacement_Z": displacementZ, 
               "Displacement_H": displacementH,
               "Stress_Z": sigmaZ,
               "Stress_R": sigmaR,
               "Stress_T": sigmaT,
               "Strain_Z": epsZ,
               "Strain_R": epsR,
               "Strain_T": epsT
               }
    return Response

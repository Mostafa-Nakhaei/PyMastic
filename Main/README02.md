  ### Input & Output Parameters
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

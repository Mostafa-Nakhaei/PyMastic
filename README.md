# PyMastic
PyMastic provides elastic response of a pavement system under uniform circular load.This open source program can be used in veriaty of tasks such pavement backcalculation, pavement design, overlay design, etc. PyMastic is written in Python programming language. The multiplatform softwere package is under construction
This program was originally written in Matlab but later transfered into Python.


## Validation examples
The PyMastic is compared against several well-esbailished multi-layer elastic programs such as WESLEA and Kenpave. The following images compare the results of Mastic with Kenpave program.

<p>
    <img src="./Images/Figure_2.png" alt align="center" height="860" width="674" >
</p>
<p>
    Figure 1. Horizontal Strain Response under circular uniform load (Kenpave VS MASTIC)
</p>



<p>
    <img src="./Images/Figure_3.png" alt align="center" height="860" width="674" >
</p>
<p>
    Figure 2. Vertical Stress Response under circular uniform load (Kenpave VS MASTIC)
</p>


<p>
    <img src="./Images/Figure_1.png" alt align="center" height="860" width="674" >
</p>
<p>
    Figure 3. Deflection of the multi layered elastic pavement under circular uniform load (Kenpave VS MASTIC)
</p>


## Pavement Deformation
The following figure shows how pavement deforms under a traffic load. The following pavement structure has 4 layers as depicted by red lines:

<p>
    <img src="./Images/Figure_4.png" alt align="center"  >
</p>
<p>
    Figure 4. Deformed Pavement Grid.
</p>

## Some extra abilities
PyMastic has the ability to change the matrix inversion method in solving boundry condition to obtain more numerically stable solution. 
The number of iteration for Henkel Intergral to converge is flexible and can be tuned by user. 

## Installation
- Clone the repository
- Run the Test.py

## Requirements
- pip install requriments.

## Sign convention:
- Positive Strain: Compressive
- Positive Stress: Compressive

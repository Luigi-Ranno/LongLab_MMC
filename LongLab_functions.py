# imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
import scipy.special as special
import math
import scipy.misc


##################################### 2D Functions
def create_array_2d(size_r, size_c, B_conc):
    #size_x/y indicate the size of the box( rows and columns).
    # B_conc indicates the concentration of B atoms(mol%)
    # the function returns an array where 0=A and 1=B
    # a good number of steps to be considered is 10* number of atoms

    box= np.zeros((size_r,size_c), dtype=np.int8)  #the initial array of pure A

    total_atoms= size_r*size_c  #total n° of atoms in the box

    B_atoms= int(np.round(total_atoms*B_conc))  # number of B atoms


    B_counter=0   #counter to place the atoms in the array


    # the while loops generates random coordinates within the box, if there are no b atoms then it assignins 1
    # possibly create lists that store the coordinates of the B atoms
    while B_counter<B_atoms:


        #randomly assigned coordinates
        r_at= rng.randint(low=0,high=size_r)
        c_at= rng.randint(low=0, high=size_c)

        # testing if a B atom is already present in that spot
        if box[r_at, c_at] != 1:

            #if the spot contains an atom A, change it with a B
            box[r_at, c_at]=1

            # increase the counter by 1
            B_counter +=1


    return box #returns the final array


def reach_equilibrium_2d(array, AB_bond, Temp, nsteps=False, evolution=False):


    # if the number of steps is not defined, then it is taken as the size of the array
    # times 100, meaning that on average each entry is analysed 100 times
    if nsteps== False:
        nsteps=  array.size *10**2

    [array_r, array_c] = np.shape(array)  #number of rows and columns in the array
    array_r -=1  #highest index that the array can have (goes from 0 to array_r-1)
    array_c -=1


    if evolution==False: # you do not want to see how the energy evolves with each step

        #iterate over the given number of steps
        for i in range(nsteps):
            # pick random atom & neighbour

            # the coordinates of the random atom
            [atom_r, atom_c] = [rng.randint(low=0, high=array_r), rng.randint(low=0, high=array_c)]


            # the coordinates of the random neighbour

            # first find the 'translating factors'
            r_factor= rng.randint(-1,1)

            # if the atom is one row lower or higher, then it must be the same column
            if r_factor== -1 or r_factor==1:
                c_factor=0

            # otherwise it can be left or right( higher/lower column)
            else:

                # to use only one expression, -1^n where n can either be 0 or 1,
                # hence the column factor can either be -1 or 1
                c_factor= (-1)**(rng.randint(0,1))

            # need to fing the row and column of the neighbour
            # this expression takes care of the edge cases without requiring an if statement
            n_r= (atom_r+r_factor)%(array_r+1)
            n_c= (atom_c + c_factor)%(array_c+1)


            # now that we have found the coordinates of the neighbours, try swapping

            #update the array once the swapping function has tried swapping them
            array=atom_swap_2d(array,atom_r,atom_c,n_r,n_c,AB_bond,Temp)

        return array # the equilibrium array, once nsteps swaps have been tried

    elif evolution==True:

        # the vector containing the number of steps followed
        steps_completed=np.arange(0,nsteps+1)

        #the vector containing the total energy of the system, first entry is the initial energy
        energy_system=[find_total_energy_2d(array, AB_bond)]


        # iterate over the given number of steps
        for i in range(nsteps):

            # pick random atom & neighbour

            # the coordinates of the random atom
            [atom_r, atom_c] = [rng.randint(low=0, high=array_r), rng.randint(low=0, high=array_c)]

            # the coordinates of the random neighbour

            # first find the 'translating factors'
            r_factor = rng.randint(-1, 1)

            # if the atom is one row lower or higher, then it must be the same column
            if r_factor == -1 or r_factor == 1:
                c_factor = 0

            # otherwise it can be left or right( higher/lower column)
            else:

                # to use only one expression, -1^n where n can either be 0 or 1,
                # hence the column factor can either be -1 or 1
                c_factor = (-1) ** (rng.randint(0, 1))

            # need to fing the row and column of the neighbour
            # this expression takes care of the edge cases without requiring an if statement
            n_r = (atom_r + r_factor) % (array_r + 1)
            n_c = (atom_c + c_factor) % (array_c + 1)

            # now that we have found the coordinates of the neighbours, try swapping

            # update the array once the swapping function has tried swapping them, get the change in energy as well
            [array, delta_E] = atom_swap_2d(array, atom_r, atom_c, n_r, n_c, AB_bond, Temp, change_e=True )

            energy_system.append(energy_system[-1] + delta_E)  #add the change in energy to the previous value
            # of the energy

        return [array,steps_completed,energy_system]  # the equilibrium array, the number of steps and the energy vectors



def calculate_energy_2d(array, r_atom, c_atom, AB_bond):
    # find the energy of a particular atom due to the bonds it formed
    # r_atom and c_atom are the coordinates of the atom whose energy is being calculated
    # the contribution between atoms of the same kind (ex A-A or B-B) is set to 0 for convenience

    [array_r, array_c] = np.shape(array)  # number of rows and columns in the array
    array_r -= 1
    array_c -= 1


    # find the different neighbours
    neighb_list= [(r_atom, (c_atom -1)%(array_c+1)),  #same row, lower columns
             (r_atom, (c_atom + 1)%(array_c+1)), #same row, higher columns
             ((r_atom -1)%(array_r+1), c_atom), #same column, lower row
             ((r_atom +1)%(array_r+1), c_atom)] #same column, higher row

    # the type of atom in the spot that is being considered( either A or B)
    atom_kind=array[r_atom,c_atom]

    different_neighb=0

    # iterating over the possible neighbours to find the atoms of a different type compared to the atom selected
    for i in neighb_list:

        if atom_kind != array[i]: # if they are of a different kind
            different_neighb +=1  # increase the counter

    # the energy equals the number of different neighbours times the AB_bond times half
    # since each bond is shared between the two atoms.

    Energy_atom= different_neighb*AB_bond*0.5

    return  Energy_atom



def atom_swap_2d(array,r1, c1, r2, c2, AB_bond, Temp, change_e=False):
    # function that decides if two atoms swap places or not
    k_B=8.617332*10**(-5)

    if change_e==False:

        #if the two atoms are different, test the swap
        if array[r1,c1] !=array[r2,c2] :

            #calculate the energy of the initial system
            E1_initial= calculate_energy_2d(array,r1,c1,AB_bond) #energy of atom in position 1, before swapping
            E2_initial= calculate_energy_2d(array, r2, c2, AB_bond) #energy of atom in position 2, before swapping
            Etot_initial= E1_initial+E2_initial #total initial energy, before swapping

            # need to record which kind of atom sits where before swapping, to avoid loosing the information
            atom_1_initial= array[r1,c1]  # the kind of atom initially sitting in position 1
            atom_2_initial= array[r2,c2]  # the kind of atom initially sitting in position 2

            #swap them
            array[r1,c1]= atom_2_initial
            array[r2,c2]= atom_1_initial
            #and calculate the energy

            E1_final= calculate_energy_2d(array,r1,c1,AB_bond)
            E2_final= calculate_energy_2d(array, r2, c2, AB_bond)
            Etot_final= E1_final + E2_final

            Delta_E=Etot_final- Etot_initial # the change in energy resulting from the swap

            #if the energy decreased, keep this configuration
            if Delta_E <= 0:
                return array

            #if the energy increased, or stayed the same, try the exp
            else:
                random_num= rng.uniform(low=0,high=1) #a uniformly distributed random number between 0 and 1

                boltzmann_factor= np.exp( -Delta_E/( k_B*Temp)  )

                if boltzmann_factor>random_num:

                    return array  #return the modified array, despite it being more energetic

                #if that failed as well, revert the array and return it
                else:
                    array[r1,c1]= atom_1_initial
                    array[r2,c2]= atom_2_initial

                    return array

        # if the two atoms are the same, changing them does not affect the system, so the initial array is
        # returned without computing anything

        else:
            return array #the input array

    elif change_e==True:  # we need to return the change in energy as well

        # if the two atoms are different, test the swap
        if array[r1, c1] != array[r2, c2]:

            # calculate the energy of the initial system
            E1_initial = calculate_energy_2d(array, r1, c1, AB_bond)  # energy of atom in position 1, before swapping
            E2_initial = calculate_energy_2d(array, r2, c2, AB_bond)  # energy of atom in position 2, before swapping
            Etot_initial = E1_initial + E2_initial  # total initial energy, before swapping

            # need to record which kind of atom sits where before swapping, to avoid loosing the information
            atom_1_initial = array[r1, c1]  # the kind of atom initially sitting in position 1
            atom_2_initial = array[r2, c2]  # the kind of atom initially sitting in position 2

            # swap them
            array[r1, c1] = atom_2_initial
            array[r2, c2] = atom_1_initial

            # and calculate the energy
            E1_final = calculate_energy_2d(array, r1, c1, AB_bond)
            E2_final = calculate_energy_2d(array, r2, c2, AB_bond)
            Etot_final = E1_final + E2_final

            Delta_E = Etot_final - Etot_initial  # the change in energy resulting from the swap

            # if the energy decreased, keep this configuration
            if Delta_E <= 0:
                return [array, Delta_E]

            # if the energy increased, or stayed the same, try the exp
            else:
                random_num = rng.uniform(low=0, high=1)  # a uniformly distributed random number between 0 and 1

                boltzmann_factor = np.exp(-Delta_E / (k_B * Temp))

                if boltzmann_factor > random_num:

                    # # ############# For debugging purposes
                    # print('This happens in the swap funcion')
                    # print('factor*100:',boltzmann_factor*100)
                    # print('random n*100:',random_num*100)
                    # print('Delta E,from swapper:',Delta_E)
                    # print('Atom in position 1 init:',atom_1_initial)
                    # print('Atom in position 1 final:',array[r1,c1])
                    # print('They should be different because the swap took place(B factor)')
                    # print('End of the swap function')
                    #
                    # # ###############


                    return [array, Delta_E]  # return the modified array, despite it being more energetic

                # if that failed as well, revert the array and return it
                else:


                    array[r1, c1] = atom_1_initial
                    array[r2, c2] = atom_2_initial

                    # ############# For debugging purposes
                    # print('factor*100:',boltzmann_factor*100)
                    # print('random n*100:',random_num*100)
                    # print('Delta E,from swapper:',Delta_E)
                    # print('Atom in position 1 init:',atom_1_initial)
                    # print('Atom in position 1 final:',array[r1,c1])
                    # print('They should be the same because the swap did not take place(B factor)')
                    #
                    # ###############

                    return [array, 0]  #no change in energy has occured

        # if the two atoms are the same, changing them does not affect the system, so the initial array is
        # returned without computing anything

        else:
            return [array, 0]  # the input array and 0(no energy change)



def analyse_array_2d(array, B_conc ):
    # need to analysed the array obtained by comparing the number of unlike neighbour with what one migh
    # expect due to randomness( modelled with a binomail distribution)

    #at first, need to iterate over the whole array to find the unlike neighbours

    number_of_atoms= array.size  #total number of atoms in the array

    [array_r, array_c] = np.shape(array)  # number of rows and columns in the array
    array_r -= 1  # highest index that the array can have (goes from 0 to array_r-1)
    array_c -= 1


    unlike_0=0  # counter for no unlike neighbours
    unlike_1=0  #             1 unlike ''
    unlike_2=0  #             2 ''''
    unlike_3=0  #             3 ''''
    unlike_4=0  #             4 ''''

    #iterate over every
    for i in range(array_r+1):
        for j in range(array_c+1):

            r_atom=i
            c_atom=j

            # find the different neighbours
            neighb_list = [[i, (j - 1)%(array_c + 1)],  # same row, lower columns
                           [i, (j + 1)%(array_c + 1)],  # same row, higher columns
                           [(i - 1)%(array_r + 1), j],  # same column, lower row
                           [(i + 1)%(array_r + 1), j]]  # same column, higher row

            # the type of atom in the spot that is being considered( either A or B)
            atom_kind = array[i, j]

            different_neighb = 0

            # iterating over the possible neighbours to find the atoms of a different type compared to the atom selected
            for k in neighb_list:

                if atom_kind != array[k[0],k[1]]:  # if they are of a different kind
                    different_neighb += 1  # increase the counter

            # depending on the number of unlike neighbours found, the respective counter is increased by 1

            if different_neighb==0:
                unlike_0 +=1

            elif different_neighb==1:
                unlike_1 +=1

            elif different_neighb==2:
                unlike_2 +=1

            elif different_neighb==3:
                unlike_3 +=1
            else:
                unlike_4 +=1


    # pack the information into a y vector
    y_measured=[unlike_0/number_of_atoms, unlike_1/number_of_atoms,
                unlike_2/number_of_atoms, unlike_3/number_of_atoms,
                unlike_4/number_of_atoms] #the entries are divided by the total
    # number of atoms to get a fraction of the total


    # now that we know the distribution of unlike neighbours in the array, the average number purely due to
    # random chance shall be calculated using a binomial distribution

    z=4 # number of neighbours in a square lattice

    y_binomial=[]  #empty vector that will be filled with the number of unlike neighbours predicted
    # by the binomial distribution

    for n in range(5): # iterating over the number of unlike neighbours

        c_n = special.factorial(z)/( special.factorial(n)*special.factorial(z-n) )

        y_binomial.append(      c_n*(  B_conc*(B_conc**(z-n))*((1-B_conc)**n) +
                                    (1-B_conc)*(B_conc**n)*((1-B_conc)**(z-n))   ) )

    y_binomial= np.array(y_binomial)
   # y_binomial= y_binomial*number_of_atoms  # convert the vector from probabilities to number of atoms that
    # are predicted to have n neighbours by multiplying by the total number of atoms

    return [y_measured,y_binomial]



def find_total_energy_2d(array,AB_bond):
    #find the total energy of the array
    #at first, need to iterate over the whole array to find the unlike neighbours

    [array_r, array_c] = np.shape(array)  # number of rows and columns in the array
    array_r -= 1  # highest index that the array can have (goes from 0 to array_r-1)
    array_c -= 1


    unlike_0=0  # counter for no unlike neighbours
    unlike_1=0  #             1 unlike ''
    unlike_2=0  #             2 ''''
    unlike_3=0  #             3 ''''
    unlike_4=0  #             4 ''''

    #iterate over every
    for i in range(array_r+1):
        for j in range(array_c+1):

            r_atom=i
            c_atom=j

            # find the different neighbours
            neighb_list = [[i, (j - 1)%(array_c + 1)],  # same row, lower columns
                           [i, (j + 1)%(array_c + 1)],  # same row, higher columns
                           [(i - 1)%(array_r + 1), j],  # same column, lower row
                           [(i + 1)%(array_r + 1), j]]  # same column, higher row

            # the type of atom in the spot that is being considered( either A or B)
            atom_kind = array[i, j]

            different_neighb = 0

            # iterating over the possible neighbours to find the atoms of a different type compared to the atom selected
            for k in neighb_list:

                if atom_kind != array[k[0],k[1]]:  # if they are of a different kind
                    different_neighb += 1  # increase the counter

            # depending on the number of unlike neighbours found, the respective counter is increased by 1

            if different_neighb==0:
                unlike_0 +=1

            elif different_neighb==1:
                unlike_1 +=1

            elif different_neighb==2:
                unlike_2 +=1

            elif different_neighb==3:
                unlike_3 +=1
            else:
                unlike_4 +=1


    #now that the unlike neighbours have been found, need to sum everything up
    unlike_0_contrib = unlike_0 * 0 * AB_bond * 0.5 #energy due to 0 unlike neighbours
    unlike_1_contrib = unlike_1 * 1 * AB_bond * 0.5 # ''1
    unlike_2_contrib = unlike_2 * 2 * AB_bond * 0.5 #''2
    unlike_3_contrib = unlike_3 * 3 * AB_bond * 0.5 #''3
    unlike_4_contrib = unlike_4 * 4 * AB_bond * 0.5 #''4

    total_energy=unlike_0_contrib + unlike_1_contrib + unlike_2_contrib + unlike_3_contrib + unlike_4_contrib
    return total_energy



def tester_2d(grid=True, energy_calc=True, swapper=True):
    # tester function used for debugging purposes

    if grid==True:  #testing if the random grid has the correct concentration of atoms

        #assign random values to the parameters that the grid generator takes
        a=rng.randint(1,20)
        b=rng.randint(1,20)
        conc= rng.uniform(0,1)

        test_array= create_array_2d(a,b,conc)
        N = a * b  # total number of atoms


        B_atoms= np.sum(test_array)  #total number of B atoms in the system (B=1, A=0)

        B_ideal= int(np.round(conc* N))


        if B_atoms != B_ideal:

            print('The array was not generated correctly')
            print('The parameter used were: a=',a, ' b=', b, ' concentration=', conc)
            print('The number of B atoms in the system is:', B_atoms, ' when it should be ', B_ideal)
            return False

        else:
            print("The function 'create_array_2d' passed the test, no bug was found")



    if energy_calc==True:  #test the function that calculates the energy of an atom

        ##  a 3x3 matrix used to test the the function

        testing_matrix= np.array( [ [0,1,0] ,  [1,0,0], [1,1,0] ] )

        AB_bond= 1

        #test that the energy at the centre is correctly calculated
        n_11=  3  # the number of unlike atoms bonded to the central atom

        E_11_ideal= n_11*1* 0.5  #only half the bond energy, the bonds are shared between 2 atoms
        E_11=calculate_energy_2d(testing_matrix, 1, 1, AB_bond)


        if E_11 != E_11_ideal:
            print(' The function did not calculate the energy correctly for the 1-1 atom')
            print('The correct energy is:', E_11_ideal,' while the function output was:',E_11 )
            return False
        else:
            print('The function correctly found the energy for the 1-1 position')


        #test that the energy at the upper left corner is correctly calculated
        n_00=  3  # the number of unlike atoms bonded to the central atom

        E_00_ideal= n_00*1* 0.5  #only half the bond energy, the bonds are shared between 2 atoms
        E_00=calculate_energy_2d(testing_matrix, 0, 0, AB_bond)


        if E_00 != E_00_ideal:
            print(' The function did not calculate the energy correctly for the 0-0 atom')
            print('The correct energy is:', E_00_ideal,' while the function output was:',E_00 )
            return False
        else:
            print('The function correctly found the energy for the 0-0 position')


        #test that the energy at the lower right corner is correctly calculated
        n_22=  2  # the number of unlike atoms bonded to the central atom

        E_22_ideal= n_22*1* 0.5  #only half the bond energy, the bonds are shared between 2 atoms
        E_22=calculate_energy_2d(testing_matrix, 2, 2, AB_bond)


        if E_22 != E_22_ideal:
            print(' The function did not calculate the energy correctly for the 2-2 atom')
            print('The correct energy is:', E_22_ideal,' while the function output was:',E_22 )
            return False
        else:
            print('The function correctly found the energy for the 2-2 position')



        #test lastly, test the bottom side of a random matrix

        #size of the matrix
        a= rng.randint(3,50)
        b= rng.randint(3,50)

        conc=0.6 #b concentration

        random_matrix= create_array_2d(a,b,conc)

        #position of a randomly chosen atom
        r= rng.randint(0,a-1)
        c= rng.randint(0,b-1)


        #find the neighbours
        if r==(a-1):
            ##at the bottom of the array

            if c== (b-1):
                # at the right

                nb = [(r - 1, c),
                      (0, c),
                      (r, c - 1),
                      (r, 0)]

            elif c==0:
                # at the left

                nb = [(r - 1, c),
                      (0, c),
                      (r, b-1),
                      (r, c + 1)]


            else:

                nb = [(r - 1, c),
                      (0, c),
                      (r, c - 1),
                      (r, c + 1)]


        elif r==0:
            # at the top of the array

            if c== (b-1):
                # at the right

                nb=[ (a-1,c),
                     (r+1,c),
                     (r,c-1),
                     (r,0)]

            elif c==0:
                # at the left

                nb=[ (a-1,c),
                     (r+1,c),
                     (r,b-1),
                     (r,c+1)]

            else:

                nb=[ (a-1,c),
                     (r+1,c),
                     (r,c-1),
                     (r,c+1)]


        else:

            #...

            if c== (b-1):
                # at the right

                nb = [(r - 1, c),
                      (r + 1, c),
                      (r, c - 1),
                      (r, 0)]

            elif c==0:

                # at the left
                nb = [(r - 1, c),
                      (r + 1, c),
                      (r, b - 1),
                      (r, c + 1)]

            else:

                nb=[ (r-1,c),
                     (r+1,c),
                     (r,c-1),
                     (r,c+1)]



        n_unlike_bond=0 #counter used to find the number of unlike bonds

        for i in nb:
            if random_matrix[i] != random_matrix[r,c]:
                n_unlike_bond += 1


        E_rng_ideal= n_unlike_bond * AB_bond * 0.5  #only half the bond energy, the bonds are shared between 2 atoms
        E_rng=calculate_energy_2d(random_matrix, r, c, AB_bond)


        if E_rng != E_rng_ideal:
            print(' The function did not calculate the energy correctly for the random matrix at position:', (r,c))
            print('The correct energy is:', E_rng_ideal,' while the function output was:', E_rng)
            print('The specific matrix that caused problems was:')
            print(random_matrix)
            return False
        else:
            print('The function correctly found the energy for an atom in a random matrix')



    if swapper == True:  #test if the atom swap function works

        #testing if the probability of swapping two atoms is compareble to the boltzmann factor

        AB_bond_to_test=0.05

        test_matrix= np.array( [[0,0,1] ,[0,0,1], [0,0,1]  ]  )

        T0=300

        #atoms to swap: (0,1) and (0,2)
        r1=0
        c1=1
        r2=0
        c2=2

        atom_1= test_matrix[r1,c1]


        #the number of unlike bonds the two atoms form are
        unlike_initial= 1+2

        #after swapping, the unlike bonds are:
        unlike_final= 4 + 3

        delta_E= AB_bond_to_test*(unlike_final - unlike_initial)*0.5 #the energy change, positive

        #the number of times the two atoms are swapped, should be proportional to e^(-dE/kT)

        P_ideal= np.exp(-delta_E/(k_B*T0))   #the ideal probability of a swap occurring


        swap_counter=0  #number of successful swaps

        trials= 10**5  #number of swaps to try


        for i in range(trials):

            # the raw array is provided because otherwise the initial array is changed
            array_to_loop= np.array(test_matrix)


            [new_array,E_change] = atom_swap_2d( np.array( array_to_loop) ,r1,c1,r2,c2,AB_bond=AB_bond_to_test,Temp=T0,change_e=True)

            if new_array[r2,c2]==atom_1:  #meaning that the swap occurred

                swap_counter += 1

                if E_change != delta_E: #when the atoms get swapped, the change in energy should match the calculated value
                    print('The change in Energy calculated by the function is wrong')
                    print('The change in energy is:',delta_E,' when the function output was:',E_change)

                    return False


            else:


                if E_change != 0:  # when the atoms get swapped, the change in energy should match the calculated value


                    print('The change in Energy calculated by the function is wrong')
                    print('The change in energy is:', 0, ' when the function output was:', E_change)

                    return False



        P_calculated= swap_counter/trials

        print('The ideal probability of swapping is:',P_ideal)
        print('The probability measured is:',P_calculated)


        accuracy=  5*10**-2

        if abs( (P_ideal-P_calculated)/P_ideal  ) > accuracy:
            print('The frational error in the two probabilities is larger than {0:.3E}, a bug is likely to be the cause'.format(accuracy))
            return False

        else:
            print('The two probabilities match within the required accuracy({0:.1E}) after {1:.1E} trials'.format(accuracy,trials))


############################### End of 2D functions


############################# 3D Functions
def create_array_3d(size_i, size_j, size_k, B_conc):
    #size_x/y/z indicate the size of the box( length height depth).
    # B_conc indicates the concentration of B atoms(mol%)
    # the function returns an array where 0=A and 1=B

    box= np.zeros((size_i,size_j, size_k), dtype=np.int8)  #the initial array of pure A

    total_atoms= box.size  #total n° of atoms in the box

    B_atoms= int(np.round(total_atoms*B_conc))  # number of B atoms


    B_counter=0   #counter to place the atoms in the array


    # the while loops generates random coordinates within the box, if there are no b atoms then it assignins 1
    # possibly create lists that store the coordinates of the B atoms
    while B_counter<B_atoms:


        #randomly assigned coordinates
        i_at= rng.randint(low=0, high=size_i)
        j_at= rng.randint(low=0, high=size_j)
        k_at= rng.randint(low=0, high=size_k)

        # testing if a B atom is already present in that spot
        if box[i_at, j_at, k_at] != 1:

            #if the spot contains an atom A, change it with a B
            box[i_at, j_at, k_at]=1

            # increase the counter by 1
            B_counter +=1


    return box #returns the final array


def reach_equilibrium_3d(array, AB_bond, Temp, nsteps=False, evolution=False):


    # if the number of steps is not defined, then it is taken as the size of the array
    # times 100, meaning that on average each entry is analysed 100 times
    if nsteps== False:
        nsteps=  array.size *10**2

    [array_i, array_j, array_k] = np.shape(array)  #number of rows and columns in the array
    array_i -=1  #highest index that the array can have (goes from 0 to array_r-1)
    array_j -=1
    array_k -=1

    if evolution==False:

        #iterate over the given number of steps
        for i in range(nsteps):
            # pick random atom & neighbour

            # the coordinates of the random atom
            [atom_i, atom_j, atom_k] = [rng.randint(low=0, high=array_i), rng.randint(low=0, high=array_j),
                                        rng.randint(low=0, high=array_k)]


            # the coordinates of the random neighbour

            # first find the 'translating factors'
            i_factor= rng.randint(-1,1)

            # if the atom is one row lower or higher, then it must be the same column
            if i_factor== -1 or i_factor==1:
                j_factor=0
                k_factor=0

            # otherwise it can be left or right or above
            else:

                #pick another number for the jth coordinate
                j_factor= rng.randint(-1,1)


                if j_factor== -1 or j_factor==1: # the neighbour is at the right or left
                    k_factor=0

                else: #the neighbour can only be above or below the atom of interest


                   # to use only one expression, -1^n where n can either be 0 or 1,
                    # hence the column factor can either be -1 or 1
                    k_factor= (-1)**(rng.randint(0,1))

            # need to fing the row and column of the neighbour
            # this expression takes care of the edge cases without requiring an if statement
            n_i= (atom_i+i_factor)%(array_i+1)
            n_j= (atom_j + j_factor)%(array_j+1)
            n_k= (atom_k + k_factor)%(array_k+1)


            # now that we have found the coordinates of the neighbours, try swapping

            #update the array once the swapping function has tried swapping them
            array=atom_swap_3d(array,atom_i,atom_j,atom_k, n_i,n_j,n_k, AB_bond,Temp)

        return array # the equilibrium array, once nsteps swaps have been tried

    elif evolution==True:

        # the vector containing the number of steps followed
        steps_completed=np.arange(0,nsteps+1)

        #the vector containing the total energy of the system, first entry is the initial energy
        energy_system=[find_total_energy_3d(array, AB_bond)]

        # iterate over the given number of steps
        for i in range(nsteps):

            # pick random atom & neighbour

            # the coordinates of the random atom
            [atom_i, atom_j, atom_k] = [rng.randint(low=0, high=array_i), rng.randint(low=0, high=array_j),
                                        rng.randint(low=0, high=array_k)]

            # the coordinates of the random neighbour

            # first find the 'translating factors'
            i_factor = rng.randint(-1, 1)

            # if the atom is one row lower or higher, then it must be the same column
            if i_factor == -1 or i_factor == 1:
                j_factor = 0
                k_factor = 0

            # otherwise it can be left or right or above
            else:

                # pick another number for the jth coordinate
                j_factor = rng.randint(-1, 1)

                if j_factor == -1 or j_factor == 1:  # the neighbour is at the right or left
                    k_factor = 0

                else:  # the neighbour can only be above or below the atom of interest


                    # to use only one expression, -1^n where n can either be 0 or 1,
                    # hence the column factor can either be -1 or 1
                    k_factor = (-1) ** (rng.randint(0, 1))

            # need to fing the row and column of the neighbour
            # this expression takes care of the edge cases without requiring an if statement
            n_i = (atom_i + i_factor) % (array_i + 1)
            n_j = (atom_j + j_factor) % (array_j + 1)
            n_k = (atom_k + k_factor) % (array_k + 1)

            # now that we have found the coordinates of the neighbours, try swapping

            # update the array once the swapping function has tried swapping them
            [array, delta_E] = atom_swap_3d(array, atom_i, atom_j, atom_k, n_i, n_j, n_k, AB_bond, Temp, change_e=True)

            energy_system.append(energy_system[-1] + delta_E)  #add the change in energy to the previous value
            # of the energy

        return [array,steps_completed,energy_system]  # the equilibrium array, the number of steps and the energy vectors


def calculate_energy_3d(array, i_atom, j_atom, k_atom, AB_bond):
    # i/j/k_atom are the coordinates of the atom whose energy is being calculated
    # the contribution between atoms of the same kind (ex A-A or B-B) is set to 0 for convenience

    [array_i, array_j, array_k] = np.shape(array)  # number of rows and columns in the array
    array_i -= 1
    array_j -= 1
    array_k -= 1


    # find the different neighbours
    neighb_list= [(i_atom, (j_atom -1)%(array_j+1), k_atom),  #same i and k, lower j
             (i_atom, (j_atom + 1)%(array_j+1), k_atom), #same i and k, higher j
             ((i_atom -1)%(array_i+1), j_atom, k_atom), #same j and k, lower i
             ((i_atom +1)%(array_i+1), j_atom, k_atom),    #same j and k, higher i
             (i_atom, j_atom, (k_atom -1)%(array_k+1) )    ,#same i and j, lower k
             (i_atom, j_atom, (k_atom + 1)%(array_k + 1))]#same i and j, higher k

    # the type of atom in the spot that is being considered( either A or B)
    atom_kind=array[i_atom, j_atom, k_atom]

    different_neighb=0

    # iterating over the possible neighbours to find the atoms of a different type compared to the atom selected
    for q in neighb_list:

        if atom_kind != array[q]: # if they are of a different kind
            different_neighb +=1  # increase the counter

    # the energy equals the number of different neighbours times the AB_bond times half
    # since each bond is shared between the two atoms.

    Energy_atom= different_neighb*AB_bond*0.5

    return  Energy_atom


def atom_swap_3d(array,i1, j1, k1, i2, j2, k2, AB_bond, Temp, change_e=False):
    # function that decides if two atoms swap places or not
    k_B=8.617332*10**(-5)

    if change_e==False:

        #if the two atoms are different, test the swap
        if array[i1, j1, k1] !=array[i2,j2,k2] :

            #calculate the energy of the initial system
            E1_initial= calculate_energy_3d(array,i1,j1,k1,AB_bond) #energy of atom in position 1, before swapping
            E2_initial= calculate_energy_3d(array, i2, j2, k2, AB_bond) #energy of atom in position 2, before swapping
            Etot_initial= E1_initial+E2_initial #total initial energy, before swapping

            # need to record which kind of atom sits where before swapping, to avoid loosing the information
            atom_1_initial= array[i1,j1, k1]  # the kind of atom initially sitting in position 1
            atom_2_initial= array[i2,j2, k2]  # the kind of atom initially sitting in position 2

            #swap them
            array[i1,j1,k1]= atom_2_initial
            array[i2,j2, k2]= atom_1_initial
            #and calculate the energy

            E1_final= calculate_energy_3d(array,i1,j1, k1,AB_bond)
            E2_final= calculate_energy_3d(array, i2, j2, k2, AB_bond)
            Etot_final= E1_final + E2_final

            Delta_E=Etot_final- Etot_initial # the change in energy resulting from the swap

            #if the energy decreased, keep this configuration
            if Delta_E< 0:
                return array

            #if the energy increased, or stayed the same, try the exp
            else:
                random_num= rng.uniform(low=0,high=1) #a uniformly distributed random number between 0 and 1

                boltzmann_factor= np.exp( -Delta_E/( k_B*Temp)  )

                if boltzmann_factor>random_num:

                    return array  #return the modified array, despite it being more energetic

                #if that failed as well, revert the array and return it
                else:
                    array[i1,j1, k1]= atom_1_initial
                    array[i2,j2, k2]= atom_2_initial

                    return array

        # if the two atoms are the same, changing them does not affect the system, so the initial array is
        # returned without computing anything

        else:
            return array #the input array

    elif change_e==True:

        # if the two atoms are different, test the swap
        if array[i1, j1, k1] != array[i2, j2, k2]:

            # calculate the energy of the initial system
            E1_initial = calculate_energy_3d(array, i1, j1, k1,
                                             AB_bond)  # energy of atom in position 1, before swapping
            E2_initial = calculate_energy_3d(array, i2, j2, k2,
                                             AB_bond)  # energy of atom in position 2, before swapping
            Etot_initial = E1_initial + E2_initial  # total initial energy, before swapping

            # need to record which kind of atom sits where before swapping, to avoid loosing the information
            atom_1_initial = array[i1, j1, k1]  # the kind of atom initially sitting in position 1
            atom_2_initial = array[i2, j2, k2]  # the kind of atom initially sitting in position 2

            # swap them
            array[i1, j1, k1] = atom_2_initial
            array[i2, j2, k2] = atom_1_initial
            # and calculate the energy

            E1_final = calculate_energy_3d(array, i1, j1, k1, AB_bond)
            E2_final = calculate_energy_3d(array, i2, j2, k2, AB_bond)
            Etot_final = E1_final + E2_final

            Delta_E = Etot_final - Etot_initial  # the change in energy resulting from the swap

            # if the energy decreased, keep this configuration
            if Delta_E < 0:
                return [array, Delta_E]

            # if the energy increased, or stayed the same, try the exp
            else:
                random_num = rng.uniform(low=0, high=1)  # a uniformly distributed random number between 0 and 1

                boltzmann_factor = np.exp(-Delta_E / (k_B * Temp))

                if boltzmann_factor > random_num:

                    return [array, Delta_E]  # return the modified array, despite it being more energetic

                # if that failed as well, revert the array and return it
                else:
                    array[i1, j1, k1] = atom_1_initial
                    array[i2, j2, k2] = atom_2_initial

                    return [array, 0]

        # if the two atoms are the same, changing them does not affect the system, so the initial array is
        # returned without computing anything

        else:
            return [array, 0]  # the input array



def analyse_array_3d(array, B_conc ):
    # need to analysed the array obtained by comparing the number of unlike neighbour with what one migh
    # expect due to randomness( modelled with a binomail distribution)

    #at first, need to iterate over the whole array to find the unlike neighbours

    number_of_atoms= array.size  #total number of atoms in the array

    [array_i, array_j, array_k] = np.shape(array)  # number of rows and columns in the array
    array_i -= 1  # highest index that the array can have (goes from 0 to array_r-1)
    array_j -= 1
    array_k -= 1


    unlike_0=0  # counter for no unlike neighbours
    unlike_1=0  #             1 unlike ''
    unlike_2=0  #             2 ''''
    unlike_3=0  #             3 ''''
    unlike_4=0  #             4 ''''
    unlike_5=0  #             5 ''''
    unlike_6=0  #             6 ''''

    #iterate over every atom
    for i in range(array_i+1):
        for j in range(array_j+1):
            for k in range(array_k+1):


                # find the different neighbours
                neighb_list = [[i, (j - 1)%(array_j + 1),k],  # same row and k, lower columns
                               [i, (j + 1)%(array_j + 1),k],  # same row and k, higher columns
                               [(i - 1)%(array_i + 1), j, k],  # same column and k, lower row
                               [(i + 1)%(array_i + 1), j, k],  # same column and k, higher row
                               [i, j, (k - 1)%(array_k+1)], # same i and j, lower k
                               [i, j, (k + 1)%(array_k+1)]] # same i and j, higher k


                # the type of atom in the spot that is being considered( either A or B)
                atom_kind = array[i, j, k]

                different_neighb = 0

                # iterating over the possible neighbours to find the atoms of a different type 
				# compared to the atom selected
                for p in neighb_list:

                    ##
                    ##



                    if atom_kind != array[p[0], p[1], p[2]]:  # if they are of a different kind
                        different_neighb += 1  # increase the counter

                # depending on the number of unlike neighbours found, the respective counter is increased by 1

                if different_neighb==0:
                    unlike_0 +=1

                elif different_neighb==1:
                    unlike_1 +=1

                elif different_neighb==2:
                    unlike_2 +=1

                elif different_neighb==3:
                    unlike_3 +=1
                elif different_neighb==4:
                    unlike_4 +=1
                elif different_neighb==5:
                    unlike_5 +=1
                else:
                    unlike_6 +=1


    #pack the information into a y vector
    y_measured=[unlike_0/number_of_atoms, unlike_1/number_of_atoms,
                unlike_2/number_of_atoms, unlike_3/number_of_atoms,
                unlike_4/number_of_atoms, unlike_5/number_of_atoms,
                unlike_6/number_of_atoms] #the entries are divided by the total
    # number of atoms to get a fraction of the total


    # now that we know the distribution of unlike neighbours in the array, the average number purely due to
    # random chance shall be calculated using a binomial distribution

    z=6 # number of neighbours in a cubic lattice

    y_binomial=[]  #empty vector that will be filled with the number of unlike neighbours predicted
    # by the binomial distribution

    for n in range(z+1): # iterating over the number of unlike neighbours

        c_n = special.factorial(z)/( special.factorial(n)*special.factorial(z-n) )

        y_binomial.append(      c_n*(  B_conc*(B_conc**(z-n))*((1-B_conc)**n) +
                                    (1-B_conc)*(B_conc**n)*((1-B_conc)**(z-n))   ) )

    y_binomial= np.array(y_binomial)
   # y_binomial= y_binomial*number_of_atoms  # convert the vector from probabilities to number of atoms that
    # are predicted to have n neighbours by multiplying by the total number of atoms

    return [y_measured,y_binomial]


def find_total_energy_3d(array,AB_bond):
    #at first, need to iterate over the whole array to find the unlike neighbours

    [array_i, array_j, array_k] = np.shape(array)  # number of rows and columns in the array
    array_i -= 1  # highest index that the array can have (goes from 0 to array_r-1)
    array_j -= 1
    array_k -= 1


    unlike_0=0  # counter for no unlike neighbours
    unlike_1=0  #             1 unlike ''
    unlike_2=0  #             2 ''''
    unlike_3=0  #             3 ''''
    unlike_4=0  #             4 ''''
    unlike_5=0  #             5 ''''
    unlike_6=0  #             6 ''''

    #iterate over every atom
    for i in range(array_i+1):
        for j in range(array_j+1):
            for k in range(array_k+1):


                # find the different neighbours
                neighb_list = [[i, (j - 1)%(array_j + 1),k],  # same row and k, lower columns
                               [i, (j + 1)%(array_j + 1),k],  # same row and k, higher columns
                               [(i - 1)%(array_i + 1), j, k],  # same column and k, lower row
                               [(i + 1)%(array_i + 1), j, k],  # same column and k, higher row
                               [i, j, (k - 1)%(array_k+1)], # same i and j, lower k
                               [i, j, (k + 1)%(array_k+1)]] # same i and j, higher k


                # the type of atom in the spot that is being considered( either A or B)
                atom_kind = array[i, j, k]

                different_neighb = 0

                # iterating over the possible neighbours to find the atoms of a different 
				# type compared to the atom selected
                for p in neighb_list:

                    ##
                    ##



                    if atom_kind != array[p[0], p[1], p[2]]:  # if they are of a different kind
                        different_neighb += 1  # increase the counter

                # depending on the number of unlike neighbours found, the respective counter is increased by 1

                if different_neighb==0:
                    unlike_0 +=1

                elif different_neighb==1:
                    unlike_1 +=1

                elif different_neighb==2:
                    unlike_2 +=1

                elif different_neighb==3:
                    unlike_3 +=1
                elif different_neighb==4:
                    unlike_4 +=1
                elif different_neighb==5:
                    unlike_5 +=1
                else:
                    unlike_6 +=1

    #now that the unlike neighbours have been found, need to sum everything up
    unlike_0_contrib = unlike_0 * 0 * AB_bond * 0.5 #energy due to 0 unlike neighbours
    unlike_1_contrib = unlike_1 * 1 * AB_bond * 0.5 # ''1
    unlike_2_contrib = unlike_2 * 2 * AB_bond * 0.5 #''2
    unlike_3_contrib = unlike_3 * 3 * AB_bond * 0.5 #''3
    unlike_4_contrib = unlike_4 * 4 * AB_bond * 0.5 #''4
    unlike_5_contrib = unlike_5 * 5 * AB_bond * 0.5 #''5
    unlike_6_contrib = unlike_6 * 6 * AB_bond * 0.5  # ''5

    total_energy=unlike_0_contrib + unlike_1_contrib + unlike_2_contrib + unlike_3_contrib + unlike_4_contrib \
                 + unlike_5_contrib + unlike_6_contrib

    return total_energy


def tester_3d(grid=True, energy_calc=True, swapper=True):
    # tester function used for debugging purposes

    if grid == True:  # testing if the random grid has the correct concentration of atoms

        # assign random values to the parameters that the grid generator takes
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(1,20)
        conc = rng.uniform(0, 1)

        test_array = create_array_3d(a, b, c, conc)
        N = a * b * c  # total number of atoms

        B_atoms = np.sum(test_array)  # total number of B atoms in the system (B=1, A=0)

        B_ideal = int(np.round(conc * N))

        if B_atoms != B_ideal:

            print('The array was not generated correctly')
            print('The parameter used were: a=', a, ' b=', b, ' c=', c, ' concentration=', conc)
            print('The number of B atoms in the system is:', B_atoms, ' when it should be ', B_ideal)
            return False

        else:
            print("The function 'create_array_3d' passed the test, no bug was found")

    if energy_calc == True:  # test the function that calculates the energy of an atom

        ##  a 3x3x3 tensor used to test the the function

        testing_matrix = np.array([  [[0, 1, 0], [1, 0, 0], [1, 1, 0]] , [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 
								   [[0, 1, 0], [1, 0, 0], [1, 1, 0]]  ])

        AB_bond = 1

        # test that the energy at the centre is correctly calculated
        n_111 = 2  # the number of unlike atoms bonded to the central atom

        E_111_ideal = n_111 * 1 * 0.5  # only half the bond energy, the bonds are shared between 2 atoms
        E_111 = calculate_energy_3d(testing_matrix, 1, 1, 1, AB_bond)

        if E_111 != E_111_ideal:
            print(' The function did not calculate the energy correctly for the 1-1 atom')
            print('The correct energy is:', E_111_ideal, ' while the function output was:', E_111)
            return False
        else:
            print('The function correctly found the energy for the 1-1-1 position')

        # test that the energy at the upper left corner is correctly calculated
        n_000 = 4  # the number of unlike atoms bonded to the central atom

        E_000_ideal = n_000 * 1 * 0.5  # only half the bond energy, the bonds are shared between 2 atoms
        E_000 = calculate_energy_3d(testing_matrix, 0, 0, 0, AB_bond)

        if E_000 != E_000_ideal:
            print(' The function did not calculate the energy correctly for the 0-0-0 atom')
            print('The correct energy is:', E_000_ideal, ' while the function output was:', E_000)
            return False
        else:
            print('The function correctly found the energy for the 0-0-0 position')

        # test that the energy at the lower right corner is correctly calculated
        n_222 = 3  # the number of unlike atoms bonded to the central atom

        E_222_ideal = n_222 * 1 * 0.5  # only half the bond energy, the bonds are shared between 2 atoms
        E_222 = calculate_energy_3d(testing_matrix, 2, 2, 2, AB_bond)

        if E_222 != E_222_ideal:
            print(' The function did not calculate the energy correctly for the 2-2-2 atom')
            print('The correct energy is:', E_222_ideal, ' while the function output was:', E_222)
            return False
        else:
            print('The function correctly found the energy for the 2-2-2 position')

        # test on a random R3 tensor

        # size of the matrix
        a = rng.randint(3, 50)
        b = rng.randint(3, 50)
        c= rng.randint(3,50)

        conc = 0.6  # b concentration

        random_matrix = create_array_3d(a, b, c, conc)

        # position of a randomly chosen atom
        i = rng.randint(0, a - 1)
        j = rng.randint(0, b - 1)
        k= rng.randint(0, c-1)




        # ###  For debugging
        # print('a,b,c:',(a,b,c))
        # print('i,j,k:',(i,j,k))
        # ###

        # find the neighbours

        if k ==(c-1):
            if i == (a - 1):
                ##at the bottom of the array

                if j == (b - 1):
                    # at the right

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i, j, k - 1),
                          (i, j, 0)]

                elif c == 0:
                    # at the left

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, b - 1, k),
                          (i, j + 1, k),
                          (i, j, k - 1),
                          (i, j, 0)]


                else:

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, j - 1, k),
                          (i, j + 1, k),
                          (i, j, k - 1),
                          (i, j, 0)]


            elif i == 0:
                # at the top of the array

                if j == (b - 1):
                    # at the right

                    nb = [(a - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i, j, k - 1),
                          (i, j, 0)]

                elif j == 0:
                    # at the left

                    nb = [(a - 1, j, k),
                          (i + 1, j, k),
                          (i, b - 1, k),
                          (i, j + 1, k),
                          (i, j, k - 1),
                          (i, j, 0)]

                else:

                    nb = [(a - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, j + 1, k),
                          (i, j, k - 1),
                          (i, j, 0)]


            else:

                # ...

                if j == (b - 1):
                    # at the right

                    nb = [(i - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i, j, k - 1),
                          (i, j, 0)]

                elif j == 0:

                    # at the left
                    nb = [(i - 1, j, k),
                          (i + 1, j, k),
                          (i, b - 1, k),
                          (i, j + 1, k),
                          (i, j, k - 1),
                          (i, j, 0)]

                else:

                    nb = [(i - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, j + 1, k),
                          (i, j, k - 1),
                          (i, j, 0)]

        elif k == 0:
            if i == (a - 1):
                ##at the bottom of the array

                if j == (b - 1):
                    # at the right

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]

                elif c == 0:
                    # at the left

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, b - 1, k),
                          (i, j + 1, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]


                else:

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, j - 1, k),
                          (i, j + 1, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]


            elif i == 0:
                # at the top of the array

                if j == (b - 1):
                    # at the right

                    nb = [(a - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]

                elif j == 0:
                    # at the left

                    nb = [(a - 1, j, k),
                          (i + 1, j, k),
                          (i, b - 1, k),
                          (i, j + 1, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]

                else:

                    nb = [(a - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, j + 1, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]


            else:

                # ...

                if j == (b - 1):
                    # at the right

                    nb = [(i - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]

                elif j == 0:

                    # at the left
                    nb = [(i - 1, j, k),
                          (i + 1, j, k),
                          (i, b - 1, k),
                          (i, j + 1, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]

                else:

                    nb = [(i - 1, j, k),
                          (i + 1, j, k),
                          (i, j - 1, k),
                          (i, j + 1, k),
                          (i, j, c - 1),
                          (i, j, k + 1)]

        else:
            if i == (a - 1):
                ##at the bottom of the array

                if j == (b - 1):
                    # at the right

                    nb = [(i - 1, j, k),
                          (0, j, k),
                          (i, j - 1, k),
                          (i, 0, k),
                          (i,j,c-1),
                          (i,j,k+1)]

                elif c == 0:
                    # at the left

                    nb = [(i - 1, j,k),
                          (0, j,k),
                          (i, b - 1,k),
                          (i, j + 1,k),
                          (i,j,c-1),
                          (i,j,k+1)]


                else:

                    nb = [(i - 1, j,k),
                          (0, j,k),
                          (i, j - 1,k),
                          (i, j + 1,k),
                          (i,j,c-1),
                          (i,j,k+1)]


            elif i == 0:
                # at the top of the array

                if j == (b - 1):
                    # at the right

                    nb = [(a - 1, j,k),
                          (i + 1, j,k),
                          (i, j - 1,k),
                          (i, 0,k),
                          (i,j,k-1),
                          (i,j,k+1)]

                elif j == 0:
                    # at the left

                    nb = [(a - 1, j,k),
                          (i + 1, j,k),
                          (i, b - 1,k),
                          (i, j + 1,k),
                          (i,j,k-1),
                          (i,j,k+1)]

                else:

                    nb = [(a - 1, j,k),
                          (i + 1, j,k),
                          (i, j - 1,k),
                          (i, j + 1,k),
                          (i,j,k-1),
                          (i,j,k+1)]


            else:

                # ...

                if j == (b - 1):
                    # at the right

                    nb = [(i - 1, j,k),
                          (i + 1, j,k),
                          (i, j - 1,k),
                          (i, 0,k),
                          (i,j,k-1),
                          (i,j,k+1)]

                elif j == 0:

                    # at the left
                    nb = [(i - 1, j,k),
                          (i + 1, j,k),
                          (i, b - 1,k),
                          (i, j + 1,k),
                          (i,j,k-1),
                          (i,j,k+1)]

                else:

                    nb = [(i - 1, j,k),
                          (i + 1, j,k),
                          (i, j - 1,k),
                          (i, j + 1,k),
                          (i,j,k-1),
                          (i,j,k+1)]

        n_unlike_bond = 0  # counter used to find the number of unlike bonds

        for q in nb:
            if random_matrix[q] != random_matrix[i, j, k]:
                n_unlike_bond += 1

        E_rng_ideal = n_unlike_bond * AB_bond * 0.5  # only half the bond energy, the bonds are shared between 2 atoms
        E_rng = calculate_energy_3d(random_matrix, i, j, k, AB_bond)

        if E_rng != E_rng_ideal:
            print(' The function did not calculate the energy correctly for the random matrix at position:', (i, j, k))
            print('The correct energy is:', E_rng_ideal, ' while the function output was:', E_rng)
            print('The specific matrix that caused problems was:')
            print(random_matrix)
            return False
        else:
            print('The function correctly found the energy for an atom in a random matrix')

    if swapper == True:  # test if the atom swap function works

        # testing if the probability of swapping two atoms is compareble to the boltzmann factor

        AB_bond_to_test = 0.01

        test_matrix = np.array( [  [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                                   [[1, 1, 1], [1, 1, 1], [1, 1, 1]] ,
                                   [[0, 0, 1], [0, 0, 1], [0, 0, 1]]])

        T0 = 300

        # atoms to swap: (0,0,1) and (1,0,1)
        i1 = 0
        j1 = 0
        k1 = 1
        i2 = 1
        j2 = 0
        k2 = 1

        atom_1 = test_matrix[i1, j1, k1]

        # the number of unlike bonds the two atoms form are
        unlike_initial = 2 + 2

        # after swapping, the unlike bonds are:
        unlike_final = 5 + 5

        delta_E = AB_bond_to_test * (unlike_final - unlike_initial) * 0.5  # the energy change, positive

        # the number of times the two atoms are swapped, should be proportional to e^(-dE/kT)

        P_ideal = np.exp(-delta_E / (k_B * T0))  # the ideal probability of a swap occurring

        swap_counter = 0  # number of successful swaps

        trials = 10 ** 5  # number of swaps to try

        for trial in range(trials):

            # the raw array is provided because otherwise the initial array is changed
            array_to_loop = np.array(test_matrix)

            [new_array, E_change] = atom_swap_3d(np.array(array_to_loop), i1, j1, k1, i2, j2, k2,
                                                 AB_bond=AB_bond_to_test, Temp=T0, change_e=True)

            if new_array[i2, j2, k2] == atom_1:  # meaning that the swap occurred

                swap_counter += 1

                if np.round(E_change) != np.round(delta_E):  # when the atoms get swapped, the change in energy should match the calculated value
                    print('The change in Energy calculated by the function is wrong')
                    print('The change in energy is:', delta_E, ' when the function output was:', E_change)

                    return False


            else:

                if E_change != 0:  # when the atoms get swapped, the change in energy should match the calculated value


                    print('The change in Energy calculated by the function is wrong')
                    print('The change in energy is:', 0, ' when the function output was:', E_change)

                    return False

        P_calculated = swap_counter / trials

        print('The ideal probability of swapping is:', P_ideal)
        print('The probability measured is:', P_calculated)

        accuracy = 5 * 10 ** -2

        if abs((P_ideal - P_calculated) / P_ideal) > accuracy:
            print(
                'The frational error in the two probabilities is larger than {0:.3E},a bug is likely to be the cause'.format(accuracy))
            return False

        else:
            print('The two probabilities match within the required accuracy({0:.1E}) after {1:.1E} trials'.format(
                accuracy, trials))


############################# End of 3D Functions


######################### Functions for WLS
def atom_swap_WLS(array, r1, c1, r2, c2, AB_bond, initial_E, ln_g, ln_g_multiplier, significant_figures=3):
    # function that decides if two atoms swap places or not
    # g is the density of states, a dictionary with a value assigned for each energy

    # if the two atoms are different, test the swap

    if array[r1, c1] != array[r2, c2]:

        # calculate the energy of the initial system
        E1_initial = calculate_energy_2d(array, r1, c1,
                                         AB_bond)  # energy of atom in position 1, before swapping
        E2_initial = calculate_energy_2d(array, r2, c2,
                                         AB_bond)  # energy of atom in position 2, before swapping
        Etot_initial = E1_initial + E2_initial  # total initial energy, before swapping

        # need to record which kind of atom sits where before swapping, to avoid loosing the information
        atom_1_initial = array[r1, c1]  # the kind of atom initially sitting in position 1
        atom_2_initial = array[r2, c2]  # the kind of atom initially sitting in position 2

        # swap them
        array[r1, c1] = atom_2_initial
        array[r2, c2] = atom_1_initial

        # and calculate the energy
        E1_final = calculate_energy_2d(array, r1, c1, AB_bond)
        E2_final = calculate_energy_2d(array, r2, c2, AB_bond)
        Etot_final = E1_final + E2_final

        Delta_E = Etot_final - Etot_initial  # the change in energy resulting from the swap
        E_swap= initial_E + Delta_E  # the energy of the array after the swap

        E_swap= np.round(E_swap,significant_figures)  #the energy of the system after swapping the atoms

        ln_g_initial = ln_g[np.round(initial_E,significant_figures)]  #density of states before swapping

        if E_swap in ln_g:  # if the energy has been visited and the density of states has a defined value
            ln_g_swap= ln_g[E_swap]

            # ###
            # print('The state has been visited')
            # ###

        else:

            # ###
            # print('New state found')
            # ###

            ln_g_swap = 0  #set the value of the density of states to 1, thus the logarithm to 0
            ln_g[E_swap] = 0

        if ln_g_swap <= ln_g_initial:  #if the density of states is lower for that swap

            #accept the swap
            ln_g[E_swap] += ln_g_multiplier  # multiply the g value for that energy by the factor

            # # ###
            # print('Swap accepted')
            # # print(ln_g)
            # # print('New ln_g:',ln_g)
            # # ###


            return [array, E_swap, ln_g]  #return the updated array, the current energy and updated density of states

        else:  # if the density of states of the swap is higher

            rand_num = rng.uniform(low=0, high=1)  #create a random number

            P_swap= math.exp(ln_g_initial - ln_g_swap)  #the probability of the unfavourable swap is set to be
            # equal to the ratio of the densities of state


            if P_swap >= rand_num:

                ln_g[E_swap] += ln_g_multiplier  # multiply the g value for that energy by the factor,
                #  so add in logarithms


                # # ###
                # print('Swap accepted')
                # # print(ln_g)
                # # print('New ln_g:', ln_g)
                # # ###



                return [ array, E_swap, ln_g]  #successful swap

            else:


                #reset the array
                array[r1, c1] = atom_1_initial
                array[r2, c2] = atom_2_initial

                ln_g[initial_E] += ln_g_multiplier  # multiply the g value for that energy by the factor

                # # ###
                # print('Swap rejected')
                # # print('New ln_g:',ln_g)
                # # ###

                return [array, initial_E, ln_g]  # no change in energy has occured
    else:

        # # ###
        # print('The same atom kind of atom was found')
        # # print('New ln_g:', ln_g)
        # # ###

        # print('ln_g_mul',ln_g_multiplier)
        # print('initial E',initial_E)

        ln_g[round(initial_E,significant_figures)] += ln_g_multiplier  # multiply the g value for that energy by the factor
        return [array, initial_E , ln_g]  # the input array and 0(no energy change)

def WLS(array, AB_bond, ln_initial_g_multiplier=1, flatness=0.2, tolerance=10**-8, counter_size=10**3, rounding_tolerance=10**-3, significant_figures=3):

    #start with a density of states with aribitrary constant value
    # to avoid floating point overflow( exponential growth due to the multiplications by the factor)
    # the logarithms are used

    [rows_array, cols_array] = np.shape(array)

    array_r = rows_array -1
    array_c = cols_array -1


    #total number of atoms
    Ntot= rows_array * cols_array

    # # the total number of states in the system is:
    # number_of_states= special.factorial( Ntot )/( special.factorial(N_Aatoms)  * special.factorial( N_Batoms)  )
    # # using Stirling's approximation formula
    #
    # log_number_of_states = -Ntot * ( (1-conc)*np.log(1-conc)  + conc*np.log(conc)  )


    #initial energy of the system
    system_initial_E= find_total_energy_2d(array, AB_bond)

    # ###
    # print('The system is starting with an energy(from WLS) of:',system_initial_E)
    # ###

    ln_g_multiplier=ln_initial_g_multiplier #setting the first value of the multiplier




    ln_g = {system_initial_E:0}  #create the logarithm of the density of states
    starting_E = np.round(system_initial_E,significant_figures) #the energy of the system used when looping

    # print('system initial E',system_initial_E)
    # print('starting E',starting_E)

    while ln_g_multiplier >= math.log(tolerance+1):  #first while, takes care of the tolerance

        H = {starting_E:1}  # the empty histogram that should be flattened

        counter=0 # number of steps in the second while loop

        flatness_achieved = 10 # assign an initial value to the flatness to initiate the while loop

        ###
        print('Current value of the ln multiplier:', ln_g_multiplier)
        print('Target:', math.log(tolerance+1))
        ###


        while flatness_achieved > flatness:  # if thee histogram is not flat enough
            ### try swapping random atoms...
            current_E= starting_E


            ##### find some random atoms in the array, and neighbours
            # the coordinates of the random atom
            [atom_r, atom_c] = [rng.randint(low=0, high=array_r), rng.randint(low=0, high=array_c)]

            # the atoms don't have to be neighbours
            [n_r, n_c]  = [rng.randint(low=0, high=array_r), rng.randint(low=0, high=array_c)]

            # # the coordinates of the random neighbour
            #
            # # first find the 'translating factors'
            # r_factor= rng.randint(-1,1)
            #
            # # if the atom is one row lower or higher, then it must be the same column
            # if r_factor== -1 or r_factor==1:
            #     c_factor=0
            #
            # # otherwise it can be left or right( higher/lower column)
            # else:
            #
            #     # to use only one expression, -1^n where n can either be 0 or 1,
            #     # hence the column factor can either be -1 or 1
            #     c_factor= (-1)**(rng.randint(0,1))
            #
            # # need to fing the row and column of the neighbour
            # # this expression takes care of the edge cases without requiring an if statement
            # n_r= (atom_r+r_factor)%(array_r+1)
            # n_c= (atom_c + c_factor)%(array_c+1)


            # now that we have found the coordinates of the neighbours, try swapping
            ######
            ###print('current E',current_E)
            [array, current_E, ln_g] = atom_swap_WLS(array, atom_r, atom_c, n_r, n_c, AB_bond, ln_g=ln_g,
                                                     initial_E=current_E, ln_g_multiplier= ln_g_multiplier,
                                                     significant_figures=significant_figures)


            ## using tolerance
            # add 1 to the bin corresponding to the energy of the updated system
            current_E = np.round(current_E, significant_figures)

            for energy in H.keys():
                if abs(current_E-energy) < tolerance:  # if the energy has been visited and
                    # the density of states has a defined value
                    H[energy] += 1
                    # print('current_E:',current_E)
                    # print('energy:',energy)
                    found= True
                    break

                else:
                    found = False

            if found == False:
                H[ current_E] =1  #create the bin


            counter += 1
            starting_E = current_E # setting the energy of for the next loop, in case
                # the histogram is flat enough

            #testing for the flatness of H
            if counter%counter_size == 0: # test for flatness after every 10^4 steps

                H_max= H[ max( H, key= H.get) ]
                H_min= H[ min( H, key= H.get) ]
                flatness_achieved = ( H_max - H_min )/ (H_max + H_min)


                # print('flatness achieved:',flatness_achieved)

                # if counter%10**5==0:  #every 10^5 steps, show the histogram
                #     print('flatness achieved:',flatness_achieved)
                #
                #     print(ln_g)
                #
                #     width=1
                #     plt.bar(H.keys(), H.values(), width, color='g')
                #     plt.show()



        ln_g_multiplier /= 2  #sqrt of the multiplier, so divide by 2 the logarithm
    return ln_g

def normalise_ln_g(array, conc, ln_g):

    # the density of states should be normalised such that integral over all energies
    # Integral_E(dg)= total number of configurations

    # if the system is small, the total number of configurations can be solved analytically
    # otherwise, need to use stirling's formula and output the logarithm of the answer( too big to be evaluated)

    [array_rows, array_cols]= array.shape

    Ntot= array_cols * array_rows #total number of atoms
    NA= int(np.round(Ntot *(1-conc)))
    NB= int(np.round(Ntot *conc))

    if Ntot < 170:  # small enough number to be evaluated

        configurations = special.factorial(Ntot)/( special.factorial(NA) * special.factorial(NB) )


        ####### to find the log of the sum of exp of logs, use logsumexp( [array] )

        log_confs = scipy.log(configurations)

        Energies = sorted( ln_g.keys() )


        values= list(ln_g.values())

        log_states_examined= scipy.misc.logsumexp( values ) #total number of states examined


        factor = log_confs - log_states_examined


        ## TODO: continue working and finish this function
        for Energy in Energies:

            ln_g[Energy] += factor

        return ln_g

    else:



        ####### to find the log of the sum of exp of logs, use logsumexp( [array] )

        log_configurations = -Ntot * (conc * np.log(conc) + (1 - conc) * np.log(1 - conc))
        Energies = sorted(ln_g.keys())

        values= list(ln_g.values())

        log_states_examined = scipy.misc.logsumexp(values)  # total number of states examined

        factor = log_configurations - log_states_examined

        ## TODO: continue working and finish this function
        for Energy in Energies:
            ln_g[Energy] += factor

        return ln_g

def Thermod(ln_g, T, N2):
    # Z=sum_E(  g(E)*e^(-E/kT) )


    elements = []

    for energy, ln_g_value in ln_g.items():
        elements.append((energy, ln_g_value))

    factor = -np.inf

    for i in range(len(elements)):

        ## find highest value to factorise
        ln_gi = elements[i][1]
        Ei = elements[i][0]
        current_value = ln_gi - (Ei / (k_B * T))

        # print('ln_gi:',ln_gi)
        # print('Ei:',Ei)
        # print('current_value:',current_value)

        if current_value > factor:
            factor = current_value


    # print('factor:',factor)
    Z_red=0  # the partition function, reduced by the factor

    log_Z= scipy.misc.logsumexp( list(ln_g.values())  )   #logarithm of the partition function

    Beta= 1/(k_B*T)   #Thermodynamic Beta

    Ev=0 # average energy

    E2v=0 # average energy^2

    for i in range(len(elements)):
        ln_gi = elements[i][1]
        Ei = elements[i][0]

        wi= math.exp(ln_gi - (Ei * Beta) - factor)  #  W= g(Ei)* e^(-Beta*Ei)

        Z_red += wi
        Ev += wi * Ei
        E2v += wi * (Ei**2)

    Ev *= 1./(Z_red )

    Cv = ( E2v / Z_red - Ev**2) * Beta/T

    return Ev, Cv/N2

#################### End of WLS Functions

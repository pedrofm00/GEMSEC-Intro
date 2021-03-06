#Torsion Angle Calculation
class Angle_Calculator:
    
    def get_backbone(work_dir, name):
        import prody as prd
        import Bio.PDB as bpdb
        
        #Parse the pdb file for its structure and the backbone
        structure = prd.parsePDB(work_dir + name + '.pdb')
        back_only = prd.writePDB(work_dir + name + "_backbone.pdb", structure.select('name N CA C'))
        
        #Parse through the backbone
        parser = bpdb.PDBParser()
        backbone = parser.get_structure(name, back_only)
        
        return backbone
    
    #Get phi/psi angles for all residues in the chain
    def get_angles(backbone):
        import Bio.PDB as bpdb
        import pandas as pd
        import numpy as np
        import multiprocessing.dummy as mp
        
        #Get phi/psi angles through biopython
        model_list = bpdb.Selection.unfold_entities(backbone, 'M')
        with mp.Pool() as pool:
            chain_list = pool.map(lambda x: x['A'], model_list)
            poly_list = pool.map(lambda x: bpdb.Polypeptide.Polypeptide(x), chain_list)
            angle_list = pool.map(lambda x: x.get_phi_psi_list(), poly_list)
            rowstuff = pool.map(lambda x: np.reshape(x,[1,len(x)*2])[0][2:-2] * (180/np.pi), angle_list)
            rowlist = list(rowstuff)
    
        #Generate labels for columns
        clmns = []
        for i in range(10):
            clmns.append('phi' f'{i+1}')
            clmns.append('psi' f'{i+1}')
        
        #Generate a dataframe that stores all phi, psi angles
        angles_by_frame = pd.DataFrame(columns = np.linspace(1,22,num = 22))
        angles_by_frame = pd.DataFrame(rowlist,index=np.linspace(1,len(rowlist),num=len(rowlist)),columns=clmns)

        #Return all angles
        return angles_by_frame
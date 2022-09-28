import torch, pickle, os
import torch_geometric as tg
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
import os.path as osp

class MTrees(Dataset):
    def __init__(self, case='test',root='../../../../tigress/cj1223/graph_merge/', ndat=1, ncube=1,  transform=None, pre_transform=None, 
                 load_path='../../../../tigress/mcranmer/merger_trees/',
                ):
        
        self.ndat=ndat
        self.ncube=ncube
        self.load_path=load_path
        self.root=root
        self.case=case
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_dir(self):
        return self.load_path
    
    @property
    def raw_file_names(self):
        return [self.root+'data_'+str(x)+'.dat' for x in range(self.ndat)]

    @property
    def processed_file_names(self):
        return [self.root+'data_'+str(x)+'.dat' for x in range(self.ndat)]

    def download(self):
        # Download to `self.raw_dir`.
        self.check_folder()
        loadhalo, loadgal=[], []
        for i in range(self.ncube):
            for j in range(self.ncube):
                for k in range(self.ncube):
                    loadhalo.append(self.load_path+f'isotrees/isotree_{i}_{j}_{k}.dat')
                    loadgal.append(self.load_path+f'/samout/{i}_{j}_{k}/galprop_0-99.dat')
        print('Got filenames')
        return loadhalo, loadgal
    
    def check_folder(self):
        cpath=osp.join(self.root, self.case)
        if not osp.exists(cpath):
            os.makedirs(cpath)
            print('Folder made')
        else:
            print(self.case+' case folder already exist')
        
    def process(self):
        m=0    
        halopaths, galpaths = self.download()
        for hp, gp in tqdm(zip(halopaths, galpaths), total=len(galpaths)):
            
            print(f'Importing data {hp}')
            pd1=pd.read_table(hp, skiprows=0, delimiter='\s+') # may want to just import target, can't enforce dtype
            raw=pd1.drop(axis=0, index=np.arange(50)).reset_index()
            trees=raw[raw.isna()['A[z]']] 
            halos=raw[~raw.isna()['A[z]']] 
            ##### I might be able to shorten this a bit #####
            f,i="float64", "int64"
            cs=[f,i,f,i,i,i,i,i,i,f,f,f,f,f,i,f,f,f,f,f,f,f,f,f,f,f]
            dicts = {}
            keys = halos.columns[1:26]
            castto = cs
            for i, key in enumerate(keys):
                    dicts[key] = castto[i]

            halos=halos.astype(dicts)
            
            ###### End ####
            print('Splitting data')
            spli=np.split(np.array(halos)[:,1:50], np.array(trees.iloc[1:].index)-np.arange(1,len(trees.index)))
            split=[]
            for s in tqdm(spli):
                if np.log10(s[0,10])>10:
                    split.append(s)
            split=np.array(split, dtype=object)

            #### Start making adjacency ####
            def convert(d,p):
                dfin=[]
                if len(p)!=len(np.unique(p)):
                    print('Wrong order of prog/desc')
                else:
                    no=d[0]
                    for desc in d:
                        if desc==no:
                            dfin.append(0)
                        else:
                            dfin.append(p.index(desc)+1)
                return dfin, np.arange(1, 1+len(p))
            ####
            print('Creating adjacency matrix')
            de, pr = [], []
            for k in tqdm(range(len(split))):
                des, pro=[], []
                for i, desc in enumerate(split[k][:,3]):
                    if desc in split[k][:i,1]:
                        des.append(int(desc))
                        pro.append(int(split[k][i,1]))
                desg, prog = convert(des,pro) #it needs to map halo 0,1,2,3 so on, not the true halo id
                de.append(desg)
                pr.append(prog)
            print(f'Adjacency matrix {m} completed')
            print(f'Loading targets')
            ##getting the galaxy output
            pdc=pd.read_table(gp, skiprows=0, delimiter=',', nrows=41, header=None)
            newcols=pdc.iloc[:,0]
            pds=pd.read_table(gp, skiprows=41, delimiter='\s+', header=None) # may want to just import target
            pds.columns=np.array(newcols)
            pd0=pds[pds[pds.columns[3]]==0.00] # subhaloes
            pdcen=pd0[(pd0[pds.columns[1]]==pd0[pds.columns[2]])] ##central haloes
            
            rhalid=np.array(pdcen[pds.columns[2]])
            halwgal=[]
            ids=[]
            out=[]
            print('Matching targets and data')
            for i, tree in enumerate(split):
                if tree[0,1] in rhalid:
                    halwgal.append(np.hstack([tree[:,[0,2]],tree[:,9:]])) #redo class so that data columns can be chosen upon initialization
                    ids.append(np.array(tree[0,1]))
                    index=np.where(rhalid==tree[0,1])
                    out.append(np.array(pdcen.iloc[index])[0][8]) #redo class so that target column(s) can be chosen upon initialization
            halwgal=np.array(halwgal, dtype=object)
            ids=np.array(ids)
            out=np.log10(np.array(out))
            dat=[]
            for n in range(len(out)):
                edge_index = torch.tensor([pro[n],
                                           des[n]], dtype=torch.long)
                x = torch.tensor(halwgal[n], dtype=torch.float)

                y=torch.tensor(out[n], dtype=torch.float)
                dat.append(Data(x=x, edge_index=edge_index, y=y, dtype=torch.float))
            print('Saving dataset')
            torch.save(dat, osp.join(self.root, self.case, f'data_{m}.pt'))
            m+=1

    def get(self, i):
        data = torch.load(dat, osp.join(self.root, self.case, f'data_{i}.pt'))
        return data
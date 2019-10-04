import pandas as pd
import numpy as np
import re
import os

def get_image_groups_by_dist(gt):
    image_names = gt.index.values.tolist()
    dict = {k: [] for k in range(3,19)}
    for filename in image_names:
        aux = int(re.search('%s(.*)%s' % ('_', 'm'), filename).group(1))
        dict[aux].append(filename)
    return dict

def load_openpose(path_to_json):
    df_openpose = pd.read_json(path_to_json)

    df_openpose=df_openpose.drop(['1', '18'], axis=1)
    df_openpose.rename(columns={'0': '1', '2': '7', '3': '9', '4': '11',
                        '5': '6', '6': '8', '7': '10', '8': '13', '9': '15',
                        '10': '17', '11': '12', '12': '14','13': '16',
                        '14': '3', '15': '2', '16': '5', '17': '4'},
                        inplace=True)
    df_openpose=df_openpose.set_index('Filename')

    return df_openpose, "coco"

def load_alphapose(path_to_json): #review this (M_4m_2) small dataset was behaving weirdly
    df = pd.read_json(path_to_json)
    df = df.sort_values(by=['score'], ascending=False)
    df = df.drop_duplicates(['image_id'])
    df = df.drop(['category_id', 'score'], axis=1)
    df = df.set_index('image_id')
    image_id=df.index.values.tolist()
    df_final = pd.DataFrame(columns = ['Filename'])
    for idx in image_id:
        aux=df.loc[idx,:]
        dict = {str(i//3 + 1) : (aux[0][i:i+2]) for i in range(0, len(aux[0]), 3)}
        dict['Filename'] = idx
        df_final = df_final.append(dict, ignore_index=True)
    return df_final.set_index('Filename'), "coco"

def load_pifpaf(path_to_json_directory):
    df_concat = pd.DataFrame()
    for file in os.listdir(path_to_json_directory):
        if file.endswith('.json'):
            df = pd.read_json(path_to_json_directory + '/' + file)
            aux=df['keypoints']
            dict = {str(i//3 + 1) : [j*4 for j in aux[0][i:i+2]] for i in range(0, len(aux[0]), 3)}
            dict['Filename'] = file[0:(-len('.pifpaf.json'))]
            df_concat=df_concat.append(dict, ignore_index=True)
    df_concat=df_concat.set_index('Filename')
    return df_concat, "coco"


def load_pose_tf(path_to_json): # the skeleton_format is not standard
    df = pd.read_json(path_to_json)
    df = df.set_index('Filename')

    for c in df.columns:
        df.loc[:,c]=pd.DataFrame(df.loc[:,c].values.tolist()).drop(2,1).values.tolist()
    df=df.rename(columns={'0': '0','1': '1','2': '2','3': '3','4': '4','5': '5',
                          '6': '10','7': '11','8': '12','9': '13','10': '14',
                          '11': '15','12': '8','13': '9'})
    return df, "mpii"

def load_gt(path_to_json):

    gt = pd.read_json(path_to_json)
    gt = gt.transpose()
    gt = gt.drop(['size'], axis=1)

    aux = pd.DataFrame(pd.DataFrame(gt['file_attributes'].tolist())['discrete pose'].tolist())
    aux.reset_index(drop=True, inplace=True)
    gt.reset_index(drop=True, inplace=True)
    gt = pd.concat([gt.drop(['file_attributes'],axis=1),aux],axis=1)
    gt = gt.set_index(['filename'])

    temp = pd.DataFrame(columns=['filename']+list(range(1,22)))
    for i in gt['regions'].iteritems():
        temp_dict = dict.fromkeys(['filename'] + list(range(1,22)))
        temp_dict['filename'] = i[0]
        for n,j in enumerate(i[1]):
            try:
                a=j['region_attributes']['Visibility']['Occluded']==True
            except:
                a=False
            temp_dict[int(j['region_attributes']['Keypoints'])]=(j['shape_attributes']['cx'],j['shape_attributes']['cy'],a)
        temp=temp.append(temp_dict , ignore_index=True)

    temp = temp.set_index(['filename'])

    #t = gt.drop(['regions'], axis=1)
    #gt_final = pd.concat([t, temp], axis=1)

    return temp

import pandas as pd
import numpy as np

def split_train_valid_pnoneside(data,train_ratio=0.8):
    """Split the data into train and valid ...

    # one smile should only emerge in one dataset(train or valid)
    """
    print("Split data in train set and valid set !")
    print("For a molecule, positive and negative sample should split into two different dataset.")
    data = data.sample(frac=1.0).reset_index(drop = True)
    smiles = list(set(data["smiles"]))
    belongs = {}
    for smile in smiles:
        belongs[smile] = int(0.5 + np.random.random())
    valid_ids = []
    train_ids = []
    for i in range(len(data)):
        smiles = data["smiles"][i]
        label = data["label"][i]
        rand = (np.random.random() < train_ratio)
        if label == belongs[smiles]:
            if rand:
                train_ids.append(i)
            else:
                valid_ids.append(i)
        else:
            if rand:
                valid_ids.append(i)
            else:
                train_ids.append(i)

    train_data = data.iloc[train_ids].sample(frac=1.0).reset_index(drop = True)
    valid_data = data.iloc[valid_ids].sample(frac=1.0).reset_index(drop = True)
    # check  here
    train_smiles = list(set(train_data["smiles"]))
    valid_smiles = list(set(valid_data["smiles"]))
    #print("Check the dataset below")
    #for i in range(1000):
    #    if train_smiles[i] not in valid_smiles:
    #        continue
    #    train_label = list(train_data[train_data["smiles"] == train_smiles[i]]["label"])
    #    valid_label = list(valid_data[valid_data["smiles"] == train_smiles[i]]["label"])
    #    if sum(train_label) + sum(valid_label) >= 20:
    #        print("Ratio : %.3f"%(1.0*sum(train_label)/(sum(valid_label) + sum(train_label))))
    print("Total samples : %d , Train samples : %d , Valid samples : %d"%(len(data),len(train_data),len(valid_data)))
    print("Total Proteins : %d , Train Proteins : %d , Valid Proteins : %d"%(
        len(set(data["protein"])),len(set(train_data["protein"])),len(set(valid_data["protein"]))
        ))
    print("Total Smiles : %d , Train Smiles : %d , Valid Smiles : %d"%(
        len(set(data["smiles"])),len(set(train_data["smiles"])),len(set(valid_data["smiles"]))
        ))
    print("Total positive ratio : %.2f , Train positive ratio : %.2f , Valid positive ratio : %.2f"%(
        len(data[data["label"] == 1])/len(data),len(train_data[train_data["label"]==1])/len(train_data),
        len(valid_data[valid_data["label"]==1])/len(valid_data)
        ))
    return train_data,valid_data


def split_train_valid_compoundoneside(data,train_ratio=0.8):
    """
    90% smiles on train side
    """
    smiles = list(set(data["smiles"]))
    np.random.shuffle(smiles)
    train_smiles = smiles[:int(train_ratio * len(smiles))]
    valid_smiles = smiles[int(train_ratio * len(smiles)):]
    train_ids = []
    valid_ids = []
    for i in range(len(data)):
        smiles = data["smiles"][i]
        if smiles in train_smiles:
            train_ids.append(i)
        else:
            valid_ids.append(i)
    train_data = data.iloc[train_ids].sample(frac=1.0).reset_index(drop = True)
    valid_data = data.iloc[valid_ids].sample(frac=1.0).reset_index(drop = True)
    # Check here
    valid_smiles = list(set(valid_data["smiles"]))
    train_smiles = list(set(train_data["smiles"]))
    for i in range(100):
        assert valid_smiles[i] not in train_smiles
    # Print Information
    print("Total samples : %d , Train samples : %d , Valid samples : %d"%(len(data),len(train_data),len(valid_data)))
    print("Total Proteins : %d , Train Proteins : %d , Valid Proteins : %d"%(
        len(set(data["protein"])),len(set(train_data["protein"])),len(set(valid_data["protein"]))
        ))
    print("Total Smiles : %d , Train Smiles : %d , Valid Smiles : %d"%(
        len(set(data["smiles"])),len(set(train_data["smiles"])),len(set(valid_data["smiles"]))
        ))
    print("Total positive ratio : %.2f , Train positive ratio : %.2f , Valid positive ratio : %.2f"%(
        len(data[data["label"] == 1])/len(data),len(train_data[train_data["label"]==1])/len(train_data),
        len(valid_data[valid_data["label"]==1])/len(valid_data)
        ))
    return train_data,valid_data

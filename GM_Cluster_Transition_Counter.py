#GMM Cluster Parsing for Count Transitions

import matplotlib.pyplot as plt

def reverse(coord_pair):
    store_1 = coord_pair[1]
    store_2 = coord_pair[4]
    rev_coords = '(' + store_1 + ', ' + store_2 + ')'
    return rev_coords

def count_unique_trans(pred):    
    transition_dict = {}

    for i in range(len(pred[:])):
        if i == len(pred[:]) - 1:
            continue
        elif pred[i+1] != pred[i]:
            key = f'{pred[i],pred[i+1]}'
            if key in transition_dict:
                transition_dict[key] += 1
            elif reverse(key) in transition_dict:
                transition_dict[reverse(key)] += 1
            else:
                transition_dict[key] = 1
    return transition_dict

def count_trans(pred):
    trans_count = 0
    
    for i in range(len(pred[:])):
        if i == len(pred[:]) - 1:
            continue
        elif pred[i+1] != pred[i]:
            trans_count += 1
    return trans_count

def transition_frequency(total, unique):
    tf = []
    for key in list(unique.keys()):
        tf.append(unique[key]/total)
    return tf

def plot_tf(trans, tf, wd, name):
    plt.close()
    plt.bar(x = list(trans.keys()), height = tf)
    plt.xticks(rotation = 75)
    plt.title('Transition Frequencies Between Clusters')
    plt.xlabel('Transition')
    plt.ylabel('Frequency')
    plt.savefig(wd + 'Transition Frequencies Between Clusters - ' + name)
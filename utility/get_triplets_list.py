import csv
import os
import pickle as pkl
import itertools

def write_triplet_list(folder_path, mode):
    print(f'\nProcessing Triplet Generation ...{folder_path}, {mode}')
    label_path = os.path.join(folder_path, f'seg_label_{mode}.pkl')
    with open(label_path, 'rb') as f:
        sample_name, labels = pkl.load(f)
    for cls in range(10):
        chosen = [(i, lab[1]) for i, lab in enumerate(labels) if lab[0] == cls] # with possible lab[2] for SLR
        print(f'number of sequences in exercise {cls}: {len(chosen)}')
        # nned to match the indices in the labels
        ind_correct = [i for i, lb in chosen if lb == 1]
        ind_incorrect = [i for i, lb in chosen if lb == 0]
        # assert len(ind_correct) == len(ind_incorrect)
        print(f'n_ce: {len(ind_correct)}, n_ie: {len(ind_incorrect)}')
        
        n_seq_per_class = min(len(ind_correct), len(ind_incorrect))
        triplet_idx = __make_triplets_combAP__(n_seq_per_class)
        triplets = [(ind_correct[a], ind_correct[p], ind_incorrect[n]) for a,p,n in triplet_idx]

        filename = f'class-{cls}_triplet_{mode}.txt'
        out_folder = os.path.join(folder_path, 'triplets')
        os.makedirs(out_folder, exist_ok=True)
        with open(os.path.join(out_folder, filename), "w") as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(triplets)
        print(f'Exercise {cls} Done\n')


def __make_triplets_insub__(n_seq_per_class):
    # n_seq_per_class: number of correct/incorrect seqs in a given exericse  
    # anchor and postive: choosing 2 from n 
    pass
    # print(f'make triplet list from {n_seq_per_class} sequences')
    # combs_ap = list(itertools.combinations(range(n_seq_per_class), 2))
    # combs = itertools.product(combs_ap, list(range(n_seq_per_class)))
    # combs = [(a,p,n) for (a,p), n in combs]
    # assert len(combs) == n_seq_per_class * (n_seq_per_class-1) /2  * n_seq_per_class
    # print(f'...and get {len(combs)} triplets: {combs[:5]}...')
    # return combs


def __make_triplets_combAP__(n_seq_per_class):
    # n_seq_per_class: number of correct/incorrect seqs in a given exericse  
    # anchor and postive: choosing 2 from n 

    print(f'make triplet list from {n_seq_per_class} sequences')
    combs_ap = list(itertools.combinations(range(n_seq_per_class), 2))
    combs = itertools.product(combs_ap, list(range(n_seq_per_class)))
    combs = [(a,p,n) for (a,p), n in combs]
    assert len(combs) == n_seq_per_class * (n_seq_per_class-1) /2  * n_seq_per_class
    print(f'...and get {len(combs)} triplets: {combs[:5]}...')
    return combs

def __make_triplets_all__(n_seq_per_class):
    # n_seq_per_class: number of correct/incorrect seqs in a given exericse  

    # return all possible combinations of (anchor, postive, negative) 
    # assert n_seq_per_class == n_achor == n_positive+1 == n_negative 

    print(f'make triplet list from {n_seq_per_class} sequences')
    all_combinations = itertools.product(range(n_seq_per_class), repeat=3)
    # Filter to include only those where b != a
    filtered_combinations = [(a, p, n) for (a, p, n) in all_combinations if p != a]
    assert len(filtered_combinations) == n_seq_per_class * (n_seq_per_class-1) * n_seq_per_class
    print(f'...and get {len(filtered_combinations)} triplets')
    return filtered_combinations


def read_triplets_list(filename):
    triplets = []
    for line in open(filename):
        a,p,n = line.split()[0], line.split()[1], line.split()[2] # strings
        triplets.append((int(a), int(p), int(n))) # anchor, close. far
    return triplets


if __name__ == '__main__':

    root = "processed_uiprmd_with_mirrors/"
    folders = [f'xsub_v{i}' for i in range(10)]
    for folder in folders:
        folder = os.path.join(root,folder)
        for split in ['train', 'val']:
            write_triplet_list(folder, split)
"""
Resolve clades to sub clades from base clades. This is complement and alternative to using clades.py
"""

import sys
from datetime import datetime
from Bio import Phylo
from Bio import SeqIO
import numpy as np

from augur.translate import construct_mut
from augur.utils import read_node_data, write_json, get_json_name, load_features
from augur.clades import is_node_in_clade, read_in_clade_definitions, get_reference_sequence_from_root_node
from augur.utils import read_metadata
import argparse
from pathlib import Path

from augur.extract_SNPs import generate_SNPs_table, make_annotation_dict, translate_feature_keep_seqs


def get_naive_direct_mutations(tree, all_muts, ref):
    direct_mutations = {}
    for node in tree.get_terminals():
        node_muts = {construct_mut(a, int(pos + 1), d) for pos, (a, d) in
                     enumerate(zip(ref['nuc'], all_muts[node.name]['sequence']))
                     if a != d and a != 'N' and d != 'N'}
        direct_mutations[node.name] = node_muts
    return direct_mutations

def get_muts_with_aln(aln, ref, debug = None):
    direct_mutations = {}
    # Compare mutations between those computed by treetime and those computed directly (still use treetime ancestral sequence)
    for seqname in aln:
            seq_muts = {construct_mut(a, int(pos+1), d) for pos, (a,d) in
                                enumerate(zip(ref['nuc'], aln[seqname]))
                      if a!=d and a != 'N' and d != 'N' }
            direct_mutations[seqname] = seq_muts
    return direct_mutations

def compare_mutations(tree, all_muts, direct_mutations):
    """
    :param tree: newick tree, needed for walking and accumulating mutations from all_muts
    :param all_muts: per node mutation dictionary from augur
    :param direct_mutations: dictionary containing direct mutation from get_muts_with_aln
    :return: message
    """
    for node in tree.get_nonterminals():
        for c in node:
            c.up = node
            c.up.clade = ""
            c.up.level = 0
            c.up.accumulated_muts = set()
    tree.root.up = None
    tree.root.sequences = {'nuc': {}}
    if 'aa_muts' in all_muts[tree.root.name]:
        tree.root.sequences.update({gene: {} for gene in all_muts[tree.root.name]['aa_muts']})

    for node in tree.find_clades(order='preorder'):
        node_muts = {}
        if node.is_terminal():
            node_muts = direct_mutations[node.name]
        if node.up:
            node.sequences = {gene: muts.copy() for gene, muts in node.up.sequences.items()}
            node.accumulated_muts = node.up.accumulated_muts.union(all_muts[node.name]['muts'])
            acc_muts = node.accumulated_muts
            if not acc_muts.issubset(node_muts) and node.is_terminal():
                print(f'Timetree accumulated mutations do not match direct mutations for {node.name}:\n'
                      f'Unique for timetree: {acc_muts.difference(node_muts)}\n'
                      f'Intersection: {acc_muts.intersection(node_muts)}\n'
                      f'Unique for direct: {node_muts.difference(acc_muts)}\n')

def resolve_clades(clade_designations, all_muts, tree, ref=None, support=10, diff=1, depth=3, debug = None):
    '''

    Ensures all nodes have an entry (or auspice doesn't display nicely), tests each node
    to see if it's the first member of a clade (assigns 'clade_annotation'), and sets
    all nodes's clade_membership to the value of their parent. This will change if later found to be
    the first member of a clade.

    Parameters
    ----------
    clade_designations :     dict
        clade definitions as :code:`{clade_name:[(gene, site, allele),...]}`
    all_muts : dict
        mutations in each node
    tree : Phylo.Tree
        phylogenetic tree to process
    ref : str/list, optional
        reference sequence to look up state when not mutated

    Returns
    -------
    dict
        mapping of node to clades
    '''

    clade_membership = {}
    direct_mutations = {}
    # parents = get_parent_name_by_child_name_for_tree(tree)

    # first pass to set all nodes to unassigned as precaution to ensure attribute is set
    for node in tree.find_clades(order='preorder'):
        clade_membership[node.name] = {'clade_membership': 'unassigned'}

    # count leaves
    for node in tree.find_clades(order='postorder'):
        node.leaf_count = 1 if node.is_terminal() else np.sum([c.leaf_count for c in node])

    for node in tree.get_nonterminals():
        for c in node:
            c.up = node
            c.up.clade = ""
            c.up.level = 0
            c.up.accumulated_muts = set()
    tree.root.up = None
    tree.root.sequences = {'nuc': {}}
    if 'aa_muts' in all_muts[tree.root.name]:
        tree.root.sequences.update({gene: {} for gene in all_muts[tree.root.name]['aa_muts']})

    # attach sequences to all nodes
    for node in tree.find_clades(order='preorder'):
        node_muts = {construct_mut(a, int(pos + 1), d) for pos, (a, d) in
                     enumerate(zip(ref['nuc'], all_muts[node.name]['sequence']))
                     if a != d and a != 'N' and d != 'N'}
        direct_mutations[node.name] = node_muts
        if node.up:
            node.sequences = {gene: muts.copy() for gene, muts in node.up.sequences.items()}
            node.accumulated_muts = node.up.accumulated_muts.union(all_muts[node.name]['muts'])
        for mut in all_muts[node.name]['muts']:
            a, pos, d = mut[0], int(mut[1:-1]) - 1, mut[-1]
            node.sequences['nuc'][pos] = d

    updated_nodes = []
    new_clades = {}
    # second pass to assign 'clade_annotation' to basal nodes within each clade
    # if multiple nodes match, assign annotation to largest
    # otherwise occasional unwanted cousin nodes get assigned the annotation
    # This assign clade to nodes that is closest to the root, therefore has the most descendants count
    basal_nodes = []
    basal_nodes_clades = {}
    for clade_name, clade_alleles in clade_designations.items():
        node_counts = []
        for node in tree.find_clades(order='preorder'):
            if is_node_in_clade(clade_alleles, node, ref):
                node_counts.append(node)
        sorted_nodes = sorted(node_counts, key=lambda x: x.leaf_count, reverse=True)
        if len(sorted_nodes) > 0:
            target_node = sorted_nodes[0]
            basal_nodes_clades[target_node.name] = clade_name
            basal_nodes.append(target_node)
            clade_membership[target_node.name] = {'clade_annotation': clade_name, 'clade_membership': clade_name}
            target_node.clade = clade_name
            target_node.level = 1

    # Third pass, resolve names for children clades
    for subtree in basal_nodes:
        for node in subtree.find_clades(order="preorder"):
            if node.name == tree.root.name and node not in basal_nodes:
                node.clade = ""
            muts = all_muts[node.name]["muts"]
            if node not in basal_nodes:
                if node.leaf_count >= support and node.name != tree.root.name and 0 < len(muts) < 100 \
                        and node.up.level < depth:
                    node.clade = node.up.clade + "/" + ",".join(muts)
                    node.level = node.up.level + 1
                    # new_clades[node.clade] = muts
                    new_clades[node.clade] = node.accumulated_muts
                elif node.up.clade:
                    node.clade = node.up.clade
                    node.level = node.up.level
                else:
                    print(f"Unexpected node: {node.name}")
                    node.clade = ",".join(muts)
            clade_membership[node.name] = {'clade_membership': node.clade}
            updated_nodes.append(node.name)
            if len(muts) >= 100:
                print(f'Excluded outliner node from clade naming: {node.name}, number of muts: {len(muts)}')
            # if node.up and node.up.level >= depth:
            #     print(f'Level execeed, node: {node.up.name}, level: {node.up.level}')

    # Fourth pass to propagate 'clade_membership'
    # don't propagate if encountering 'clade_annotation'
    for node in tree.find_clades(order='preorder'):
        for child in node:
            if child.name in clade_membership and 'clade_annotation' not in clade_membership[child.name] and \
                    node.name not in updated_nodes:
                clade_membership[child.name]['clade_membership'] = clade_membership[node.name]['clade_membership']

    return clade_membership, new_clades, direct_mutations


def _find_neighbors(dict_muts, query, max_dist=1):
    neighbors = set()
    for key in dict_muts:
        distance = len(set(dict_muts[query]) ^ set(dict_muts[key]))
        if distance <= max_dist:
            neighbors.add(key)
    return neighbors


def find_clusters(dict_muts, metadata, max_dist=1, min_support=10):
    """
    Implementation following the classic DBSCAN algorithm.
    Find clusters of sequences that have maximum max_dist away from each other.

    Parameters
        dict_muts: dict dictionary with keys as strain names, values as mutations of the strain against same reference

    Returns
        clusters: 2d dict cluster no. as keys;
    """
    clusters = {}
    processed = {}
    count = 0
    core_points = {}
    for tip in dict_muts:
        if tip in processed: continue
        neighbors = _find_neighbors(dict_muts, tip, max_dist)
        if len(neighbors) < min_support:
            processed[tip] = "outlier"
            continue
        count += 1 # next cluster label
        processed[tip] = count
        seed_set = neighbors.difference(set(tip))
        clusters[count] = {}
        core_points[count] = [tip]
        full_set = seed_set.copy()
        for item in seed_set:
            if item in processed:
                if processed[item] == "outlier": processed[item] = count  # border point
                else:
                    continue  # previously processed border point
            processed[item] = count  # asign cluster for item
            neighbors = _find_neighbors(dict_muts, item, max_dist)
            if len(neighbors) >= min_support:  # density check, (if item is a core point
                full_set.update(neighbors)
                core_points[count].append(item)
        clusters[count]['set'] = full_set
    # select core point based on oldest date.
    for cluster in core_points:
        sort_by_date = sorted(core_points[cluster], key = lambda x: datetime.strptime(metadata[x], "%Y-%m-%d"))
        clusters[cluster]['core'] = sort_by_date[0]
        clusters[cluster]['date'] = metadata[sort_by_date[0]]
    return clusters, processed


def register_arguments(parser):
    parser.add_argument('-t', '--tree', help="prebuilt Newick -- no tree will be built if provided")
    parser.add_argument('-m', '--mutations', nargs='+',
                        help='JSON(s) containing ancestral and tip nucleotide and/or amino-acid mutations ')
    parser.add_argument('-r', '--reference',
                        help='reference sequence in genbank format which was used for alignment')

    parser.add_argument('-a', '--alignment', help="alignment in fasta")
    parser.add_argument('-c', '--clades', type=str, help='TSV file containing clade definitions')
    parser.add_argument('--metadata', type=str, required= True, metavar="FILE", help='TSV file containing metadata, with at least strain names and dates')
    parser.add_argument('-n', '--new-clades', type=str, help='Write out TSV file containing new clades')
    parser.add_argument('--clade-assignment', '--ca', type=str, help='Write out clade assignment')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Maximum depth to search for sub-clades from base clade')
    parser.add_argument('--min-support', type=int, default=10,
                        help='Mininum number of sequences to form a new sub-clade')
    parser.add_argument('--min-mutation', type=int, default=1,
                        help='Minimum number of mutations compare to the most recent ancestral sequence to form a new clade. Defunct, not using')
    parser.add_argument('--output-node-data', '--ond', type=str, help='name of JSON file to save clade assignments to')
    parser.add_argument('--output-tip-cluster', type=str, metavar="JSON", help="output name JSON tip clusters")
    parser.add_argument('--refid', help="Reference sequence ID in the alignment file in case there is a mismatch between reference sequence and alignment")


def run(args):
    ## read tree and data, if reading data fails, return with error code
    tree = Phylo.read(args.tree, 'newick')
    node_data = read_node_data(args.mutations, args.tree)
    if node_data is None:
        print("ERROR: could not read node data (incl sequences)")
        return 1
    all_muts = node_data['nodes']

    # extract reference sequences from the root node entry in the mutation json
    # if this doesn't exist, it will complain but not error.
    inferred_root = get_reference_sequence_from_root_node(all_muts, tree.root.name)

    min_support = args.min_support
    min_mutation = args.min_mutation
    clade_designations = read_in_clade_definitions(args.clades)
    refseq = {}
    refseq['nuc'] = list(SeqIO.read(args.reference, 'genbank').seq)
    # assert(refseq['nuc'] == "".join(inferred_root['nuc'])) # Suppose to fail

    from Bio import AlignIO
    alignment = AlignIO.read(open(args.alignment), "fasta")
    aln_seq = {}
    for s in alignment:
        aln_seq[s.id] = s.seq

        ## check file format and read in sequences
    from Bio import AlignIO
    alignment = AlignIO.read(open(args.alignment), "fasta")
    seq_dict = {}
    refid = SeqIO.read(args.reference, 'genbank').id
    if args.refid:
        refid = args.refid

    for r in alignment:
        if r.id == refid:
            seq_dict['refseq'] = r.seq
        else:
            seq_dict[r.id] = r.seq

    ## load features; only requested features if genes given
    features = load_features(args.reference, None)
    print("Read in {} features from reference sequence file".format(len(features)))
    if features is None:
        print("ERROR: could not read features of reference sequence file")
        return 1

    ### translate every feature - but not 'nuc'!
    translations = {}
    for fname, feat in features.items():
        if feat.type != 'source':
            translations[fname] = translate_feature_keep_seqs(seq_dict, feat)

    annotations = make_annotation_dict(features, args.reference)

    ## determine amino acid mutations for each sequence
    seq_ids = [k for k in seq_dict.keys() if k != 'refseq']
    aa_muts = generate_SNPs_table(seq_ids, translations, annotations)


    metadata, _ = read_metadata(args.metadata)
    strain2date = {strain:metadata[strain]['date'] for strain in metadata}

    clade_membership, new_clades, _ = resolve_clades(clade_designations, all_muts, tree, inferred_root,
                                                                    support=min_support, diff=min_mutation,
                                                                    depth=args.max_depth)
    direct_mutations = get_muts_with_aln(aln_seq, refseq)
    # compare_mutations(tree, all_muts, direct_mutations)
    # sort direct mutations
    direct_mutations = {node: sorted(list(direct_mutations[node]), key=lambda x: int(x[1:-1])) for node in direct_mutations}
    clusters,strains2clusters = find_clusters(direct_mutations, strain2date, max_dist= 1, min_support= min_support)
    if args.new_clades:
        clade_out = open(args.new_clades, 'w')
        for clade, muts in new_clades.items():
            for mut in muts:
                clade_out.write(f'{clade}\tnuc\t{mut[1:-1]}\t{mut[-1]}\n')
        print(f"Wrote new clades to: {args.new_clades}")
        clade_out.close()
    out_name = get_json_name(args)
    write_json({'nodes': clade_membership}, out_name)
    print("clades written to", out_name, file=sys.stderr)
    clade_assignment = args.clade_assignment if args.clade_assignment else Path(out_name).parent.joinpath(
        "clade_assignment.tsv")

    print("Clustering sequences")

    # clusters = {node: [] for node in tip_direct_mutations}

    with open(clade_assignment, "w") as cah:
        cah.write("strain\tclade\tcluster_core\tcluster_size\tcluster_date\tdirect_mutations\tdirect_aa_mutations\n")
        for strain in seq_ids:
            dmuts = direct_mutations[strain]
            aa_dmuts = aa_muts[strain]['aa']

            # TODO: Check if the following segment can be more idiomatic
            ga_muts = {}
            for v in aa_dmuts.values():
                mutstr = f"{v['ref']}{v['p']}{v['alt']}"
                if v['g'] in ga_muts:
                    ga_muts[v['g']].append(mutstr)
                else:
                    ga_muts[v['g']] = [mutstr]
            amutstr = [k + ":" + ','.join(v) for k, v in ga_muts.items()]
            amutstr = ';'.join(amutstr)

            cluster_name = strains2clusters[strain]
            cluster_core = "undefined"
            cluster_size = "unkown"
            cluster_date = "unkown"
            if cluster_name in clusters:
                cluster_core = clusters[cluster_name]['core']
                cluster_size = len(clusters[cluster_name]['set'])
                cluster_date = clusters[cluster_name]['date']
            membership = clade_membership[strain]['clade_membership']
            cah.write( f"{strain}\t{membership}\t{cluster_core}\t{cluster_size}\t{cluster_date}\t{','.join(dmuts)}\t{amutstr}\n")
    print("clade assignments written to", clade_assignment, file=sys.stderr)


    node_data = {}
    for n in tree.find_clades(order='postorder'):
        if n.is_terminal():
            cluster_name = strains2clusters[n.name]
            cluster_core = "undefined"
            if cluster_name in clusters:
                   cluster_core = clusters[cluster_name]['core']
            node_data[n.name] = {'tip_cluster':cluster_core}

    tip_cluster_out  = args.output_tip_cluster if args.output_tip_cluster else Path(out_name).parent.joinpath("tip_cluster.json")
    write_json({"nodes":node_data}, tip_cluster_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clades-resolve",
        description="Attempt to resolve clades for sequences at higher resolution, also print out new clade.tsv")
    args = register_arguments(parser)
    exit(run(parser.parse_args()))

"""
Resolve clades to sub clades from base clades. This is complement and alternative to using clades.py
"""

import sys
from Bio import Phylo
import pandas as pd
import numpy as np
from collections import defaultdict

from augur.translate import construct_mut
from augur.utils import get_parent_name_by_child_name_for_tree, read_node_data, write_json, get_json_name
from augur.clades import is_node_in_clade, read_in_clade_definitions, get_reference_sequence_from_root_node
import argparse
from pathlib import Path


def get_naive_direct_mutations(tree, all_muts, ref):
    direct_mutations = {}
    for node in tree.get_terminals():
        node_muts = {construct_mut(a, int(pos + 1), d) for pos, (a, d) in
                     enumerate(zip(ref['nuc'], all_muts[node.name]['sequence']))
                     if a != d and a != 'N' and d != 'N'}
        direct_mutations[node.name] = node_muts
    return direct_mutations


def resolve_clades(clade_designations, all_muts, tree, ref=None, support=10, diff=1, depth=3):
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
        node_muts = {construct_mut(a, int(pos+1), d) for pos, (a,d) in
                            enumerate(zip(ref['nuc'], all_muts[node.name]['sequence']))
                     if a!=d and a != 'N' and d != 'N' }
        direct_mutations[node.name] = node_muts
        if node.up:
            node.sequences = {gene: muts.copy() for gene, muts in node.up.sequences.items()}
            node.accumulated_muts = node.up.accumulated_muts.union(all_muts[node.name]['muts'])
            acc_muts = node.accumulated_muts
            if not acc_muts.issubset(node_muts) and node.is_terminal():
               print(f'Timetree accumulated mutations do not match direct mutations for {node.name}:\n'
                     f'Unique for timetree: {acc_muts.difference(node_muts)}\n'
                     f'Intersection: {acc_muts.intersection(node_muts)}\n'
                     f'Unique for direct: {node_muts.difference(acc_muts)}\n')
               nn = node.name
               if nn == "Hungary/SRC-00817/2020":
                    p = [n.name for n in tree.get_path(node)]
                    tmp = {n: all_muts[n]['muts'] for n in p }
                    print(f"Pause debug: {','.join(p)}\n")
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


def register_arguments(parser):
    parser.add_argument('-t', '--tree', help="prebuilt Newick -- no tree will be built if provided")
    parser.add_argument('-m', '--mutations', nargs='+',
                        help='JSON(s) containing ancestral and tip nucleotide and/or amino-acid mutations ')
    parser.add_argument('-r', '--reference', nargs='+',
                        help='fasta files containing reference and tip nucleotide and/or amino-acid sequences ')
    parser.add_argument('-c', '--clades', type=str, help='TSV file containing clade definitions')
    parser.add_argument('-n', '--new-clades', type=str, help='Write out TSV file containing new clades')
    parser.add_argument('--clade-assignment', '--ca', type=str, help='Write out clade assignment')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Maximum depth to search for sub-clades from base clade')
    parser.add_argument('--min-support', type=int, default=10,
                        help='Mininum number of sequences to form a new sub-clade')
    parser.add_argument('--min-mutation', type=int, default=1,
                        help='Minimum number of mutations compare to the most recent ancestral sequence to form a new clade. Defunct, not using')
    parser.add_argument('--output-node-data', '--ond', type=str, help='name of JSON file to save clade assignments to')


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
    ref = get_reference_sequence_from_root_node(all_muts, tree.root.name)

    min_support = args.min_support
    min_mutation = args.min_mutation
    clade_designations = read_in_clade_definitions(args.clades)

    clade_membership, new_clades, direct_mutations = resolve_clades(clade_designations, all_muts, tree, ref,
                                                  support=min_support, diff=min_mutation, depth=args.max_depth)
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
    write_tip_only = True
    with open(clade_assignment, "w") as cah:
        cah.write("strain\tclade\tmutations\tdirect_mutations\n")
        if write_tip_only:
            for node in tree.get_terminals():
                if node.name in clade_membership:
                    node_path = tree.get_path(node)
                    muts = []
                    dmuts = sorted(list(direct_mutations[node.name]), key=lambda x: int(x[1:-1]))
                    for n in node_path:
                        muts += all_muts[n.name]['muts']

                    muts = sorted(muts, key=lambda x: int(x[1:-1]))
                    cah.write(f"{node.name}\t{clade_membership[node.name]['clade_membership']}\t{','.join(muts)}\t{','.join(dmuts)}\n")
        else:
            for node in tree.find_clades(order='preorder'):
                if node.name in clade_membership:
                    cah.write(
                        f"{node.name}\t{clade_membership[node.name]['clade_membership']}\t{','.join(all_muts[node.name]['muts'])}\n")
    print("clade assignments written to", clade_assignment, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clades-resolve",
        description="Attempt to resolve clades for sequences at higher resolution, also print out new clade.tsv")
    args = register_arguments(parser)
    exit(run(parser.parse_args()))

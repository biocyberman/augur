"""
Resolve clades to sub clades from base clades. This is complement and alternative to using clades.py
"""

import sys
from Bio import Phylo
import pandas as pd
import numpy as np
from collections import defaultdict
from augur.utils import get_parent_name_by_child_name_for_tree, read_node_data, write_json, get_json_name
from augur.clades import is_node_in_clade, read_in_clade_definitions, get_reference_sequence_from_root_node
import argparse

def resolve_clades(clade_designations, all_muts, tree, ref=None, support = 10, diff = 3):
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
    parents = get_parent_name_by_child_name_for_tree(tree)

    # first pass to set all nodes to unassigned as precaution to ensure attribute is set
    for node in tree.find_clades(order = 'preorder'):
        clade_membership[node.name] = {'clade_membership': 'unassigned'}

    # count leaves
    for node in tree.find_clades(order = 'postorder'):
        node.leaf_count = 1 if node.is_terminal() else np.sum([c.leaf_count for c in node])

    for node in tree.get_nonterminals():
        for c in node:
            c.up=node
    tree.root.up = None
    tree.root.sequences = {'nuc':{}}
    if 'aa_muts' in all_muts[tree.root.name]:
        tree.root.sequences.update({gene:{} for gene in all_muts[tree.root.name]['aa_muts']})

    # attach sequences to all nodes
    for node in tree.find_clades(order='preorder'):
        if node.up:
            node.sequences = {gene:muts.copy() for gene, muts in node.up.sequences.items()}
        for mut in all_muts[node.name]['muts']:
            a, pos, d = mut[0], int(mut[1:-1])-1, mut[-1]
            node.sequences['nuc'][pos] = d
        if 'aa_muts' in all_muts[node.name]:
            for gene in all_muts[node.name]['aa_muts']:
                for mut in all_muts[node.name]['aa_muts'][gene]:
                    a, pos, d = mut[0], int(mut[1:-1])-1, mut[-1]

                    if gene not in node.sequences:
                        node.sequences[gene]={}
                    node.sequences[gene][pos] = d

    updated_nodes = []
    def _update_clade(clade_name, clade_alleles, subtree, clade_level):
        if subtree is not None and clade_level <= 6:
            node_counts = []
            for node in subtree:
                if is_node_in_clade(clade_alleles, node, ref):
                    node_counts.append(node)
            sorted_nodes = sorted(node_counts, key=lambda x: x.leaf_count, reverse=True)
            alleles = clade_alleles
            if len(sorted_nodes) > 0:
                target_node = sorted_nodes[0]
                muts = all_muts[target_node.name]['muts']
                mut_str = ','.join(muts)
                cl_name = clade_name
                level = clade_level
                if len(muts) > 0 and target_node.leaf_count >= support:
                    cl_name = clade_name + '/' + mut_str
                    if clade_level > 2:
                        print(f"Clade: {cl_name}, level: {clade_level}, support: {target_node.leaf_count}")
                    if target_node.name not in updated_nodes:
                        clade_membership[target_node.name] = {'clade_annotation': cl_name, 'clade_membership': cl_name}
                        updated_nodes.append(target_node.name)

                for mut in all_muts[node.name]['muts']:
                    a, pos, d = mut[0], int(mut[1:-1]) - 1, mut[-1]
                    allele = ('nuc', pos, d)
                    alleles.append(allele)
                level += 1
                _update_clade(cl_name, alleles, target_node, level)
            else:
                _update_clade(None, None, None, None)
        else:
            return

    # Second pass to assign 'clade_annotation' to basal nodes within each clade
    # if multiple nodes match, assign annotation to largest
    # otherwise occasional unwanted cousin nodes get assigned the annotation
    for clade_name, clade_alleles in clade_designations.items():
        node_counts = []
        for node in tree.find_clades(order = 'level'):
           _update_clade(clade_name, clade_alleles, node, clade_level= 0)

    # third pass to propagate 'clade_membership'
    # don't propagate if encountering 'clade_annotation'
    for node in tree.find_clades(order = 'preorder'):
        for child in node:
            if child.name in clade_membership and 'clade_annotation' not in clade_membership[child.name]:
                clade_membership[child.name]['clade_membership'] = clade_membership[node.name]['clade_membership']

    return clade_membership

def register_arguments(parser):
    parser.add_argument('--tree', help="prebuilt Newick -- no tree will be built if provided")
    parser.add_argument('--mutations', nargs='+', help='JSON(s) containing ancestral and tip nucleotide and/or amino-acid mutations ')
    parser.add_argument('--reference', nargs='+', help='fasta files containing reference and tip nucleotide and/or amino-acid sequences ')
    parser.add_argument('--clades', type=str, help='TSV file containing clade definitions by amino-acid')
    parser.add_argument('--output-node-data', type=str, help='name of JSON file to save clade assignments to')


def run(args):
    ## read tree and data, if reading data fails, return with error code
    tree = Phylo.read(args.tree, 'newick')
    node_data = read_node_data(args.mutations, args.tree)
    if node_data is None:
        print("ERROR: could not read node data (incl sequences)")
        return 1
    all_muts = node_data['nodes']

    if args.reference:
        # PLACE HOLDER FOR vcf WORKFLOW.
        # Works without a reference for now but can be added if clade defs contain positions
        # that are monomorphic across reference and sequence sample.
        ref = None
    else:
        # extract reference sequences from the root node entry in the mutation json
        # if this doesn't exist, it will complain but not error.
        ref = get_reference_sequence_from_root_node(all_muts, tree.root.name)

    clade_designations = read_in_clade_definitions(args.clades)

    clade_membership = resolve_clades(clade_designations, all_muts, tree, ref, support= 15)

    out_name = get_json_name(args)
    write_json({'nodes': clade_membership}, out_name)
    print("clades written to", out_name, file=sys.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clades-resolve",
        description="Attempt to resolve clades for sequences at higher resolution, also print out new clade.tsv")
    args = register_arguments(parser)
    exit(run(parser.parse_args()))

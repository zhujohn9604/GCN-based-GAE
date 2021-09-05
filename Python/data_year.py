import os
import itertools as it

from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np

from collections import namedtuple

# We define a typed node such that key is garanteed to be unique within type
TypedNode = namedtuple('TypedNode', ['type', 'key'])

AUTHOR_TYPE = 'author'
PAPER_TYPE = 'paper'
SUBJECT_TYPE = 'subject'

graph_dir = './'

def load_tables(graph_dir, target_start_year, target_end_year, with_authors=True, low_memory=False):
    """ Load tables as extracted by harvesting scripts """
    print("Loading graph from tables in:", graph_dir)
    print("Loading patents...")
    df_paper = pd.read_csv(os.path.join(graph_dir, 'patent.csv'), low_memory=low_memory)
    df_paper = df_paper[(df_paper.appln_filing_year <= target_end_year) & (target_start_year <= df_paper.appln_filing_year)].reset_index(drop=True)
    print("N patents", len(df_paper))
    print("Setting idx to first column")
    # Cast to string first
    df_paper[df_paper.columns[0]] = df_paper[df_paper.columns[0]].astype(str)
    # Set index inplace and drop old column
    # Important to access column 0 by its name
    df_paper.set_index(df_paper.columns[0], drop=True, inplace=True)
    df_paper.index = df_paper.index.map(int)
    print("Index type:", df_paper.index.dtype)
    print("Loading annotations...")
    df_annotation = pd.read_csv(os.path.join(graph_dir, 'annotation.csv'), low_memory=low_memory)
    df_annotation = df_annotation[(df_annotation.appln_filing_year <= target_end_year) & (target_start_year <= df_annotation.appln_filing_year)].reset_index(drop=True)
    print("N annotations", len(df_annotation))
    if not with_authors:
        return df_paper, df_annotation

    # Else, also load authors
    print("Loading authors...")
    df_author = pd.read_csv(os.path.join(graph_dir, 'authorship.csv'),
                            dtype={'paper_id': df_paper.index.dtype, 'author': str},
                            low_memory=low_memory)
    print(df_author.head())

    print("N authorships", len(df_author))
    return df_paper, df_annotation, df_author


class BiblioGraph:
    def __init__(self, graph, paper_features, paper_labels=None):
        self.graph = graph
        # Paper has additional info we need to store
        self.paper_features = paper_features
        self.paper_labels = paper_labels  # None if unsupervised
        self.is_numeric = False
        self.is_supervised = paper_labels is not None

        # features available after numericalize
        self.ndata = None

    def __str__(self):
        if not self.is_numeric:
            return "BiblioGraph with {} nodes ({} papers) and {} edges." \
                .format(self.graph.number_of_nodes(),
                        len(self.paper_features),
                        self.graph.number_of_edges())
        else:
            n_authors = len(self.ndata[self.ndata.type == AUTHOR_TYPE])
            n_papers = len(self.ndata[self.ndata.type == PAPER_TYPE])
            n_subjects = len(self.ndata[self.ndata.type == SUBJECT_TYPE])
            avg_degree = np.mean(list(zip(*self.graph.degree))[1])
            return "BiblioGraph with {} nodes ({} authors, {} papers, {} subjects) and {} edges, average degree: {}." \
                .format(self.graph.number_of_nodes(),
                        n_authors,
                        n_papers,
                        n_subjects,
                        self.graph.number_of_edges(),
                        avg_degree)

    def save_cache(self, cache_dir):
        """ Saves a numerical representation of a graph to some cache"""
        if not self.is_numeric:
            raise ValueError("No need to save if not numeric")
        if self.is_supervised:
            raise NotImplementedError("Caching not yet implemented for supervised views")

        os.makedirs(cache_dir, exist_ok=True)
        # Adjacencies
        adj_path = os.path.join(cache_dir, "adjlist.txt")
        nx.readwrite.adjlist.write_adjlist(self.graph, adj_path)
        ndata_path = os.path.join(cache_dir, "ndata.csv")
        self.ndata.to_csv(ndata_path, index=True)

    @staticmethod
    def load_cache(cache_dir, paper_features_path, low_memory=False):
        """ Load a cached, numerical representation of a graph """
        # Adjacencies
        adj_path = os.path.join(cache_dir, "adjlist.txt")
        g = nx.readwrite.adjlist.read_adjlist(adj_path)
        paper_features = pd.read_csv(paper_features_path, low_memory=low_memory)
        paper_features.set_index(paper_features[paper_features.columns[0]].astype(str),
                                 inplace=True)
        bg = BiblioGraph(g, paper_features)

        # Node data including types and stuff
        ndata = pd.read_csv(os.path.join(cache_dir, "ndata.csv"), index_col=0,
                            dtype={"identifier": str,
                                   "type": str},
                            low_memory=low_memory)

        print("Ndata identifier dtype", ndata.identifier.dtype)
        print("Paper features index dtype", paper_features.index.dtype)

        reindexed_paper_features = ndata[ndata.type == PAPER_TYPE].join(paper_features,
                                                                        on="identifier",
                                                                        how="inner")
        bg.ndata = ndata
        bg.paper_features = reindexed_paper_features
        bg.is_numeric = True
        bg.is_supervised = False
        bg.paper_labels = None
        return bg

    @staticmethod
    def from_tables(df_paper, df_annotation, df_author=None,
                    undirected=True, supervised=False, collate_coauthorship=False):

        graph_cls = nx.Graph if undirected else nx.DiGraph
        g = graph_cls()

        print("Adding patent nodes")
        for row in tqdm(df_paper.itertuples(index=True)):
            paper = TypedNode(PAPER_TYPE, row.Index)
            g.add_node(paper, title=row.appln_abstract, year=row.appln_filing_year)

        if df_author is not None:
            if not collate_coauthorship:
                print("Inserting authorship edges")
                for paper_id, author_id in tqdm(df_author.itertuples(index=False)):
                    paper = TypedNode(PAPER_TYPE, paper_id)
                    author = TypedNode(AUTHOR_TYPE, author_id)
                    g.add_edge(author, paper)
            else:
                print("Collating coauthorship edges between papers")
                for __author, group in tqdm(df_author.groupby('author')):
                    coauthored_papers = group['paper_id'].values
                    for p1, p2 in it.product(coauthored_papers, coauthored_papers):
                        p1, p2 = TypedNode(PAPER_TYPE, p1), TypedNode(PAPER_TYPE, p2)
                        g.add_edge(p1, p2)

        if not supervised:
            print("Inserting annotation edges")
            for row in tqdm(df_annotation.itertuples(index=True)):
                paper = TypedNode(PAPER_TYPE, row.appln_id)
                subject = TypedNode(SUBJECT_TYPE, row.label)
                g.add_edge(paper, subject)

        else:
            raise NotImplementedError("Supervised loading not yet implemented")

        return BiblioGraph(g, df_paper)

    def get_identifiers(self, nodetype):
        return self.ndata[self.ndata.type == nodetype]["identifier"].values

    def get_mask(self, nodetype):
        return (self.ndata.type == nodetype).values

    def numericalize_(self):
        if self.is_numeric:
            raise ValueError("Graph is already numeric")
        print("Converting node labels to integers")
        g = nx.convert_node_labels_to_integers(self.graph, label_attribute='label')
        node_index, node_types, node_identifiers = [], [], []
        for node, node_data in g.nodes(data=True):
            node_type, node_identifier = node_data['label']
            node_types.append(node_type)
            node_identifiers.append(node_identifier)
            node_index.append(node)

        ndata = pd.DataFrame({"type": node_types, "identifier": node_identifiers}, index=node_index)

        old_n = len(self.paper_features)
        reindexed_paper_features = ndata[ndata.type == PAPER_TYPE].join(self.paper_features,
                                                                        on='identifier',
                                                                        how="inner")
        self.ndata = ndata
        self.paper_features = reindexed_paper_features
        assert len(self.paper_features) == old_n
        assert len(self.ndata[self.ndata.type == PAPER_TYPE]) == len(self.paper_features)
        self.graph = g
        self.is_numeric = True
        return self

    def label_indicator_matrix(self, dtype=np.uint8):
        if not self.is_numeric:
            raise ValueError("Graph not numeric. Call numericalize_() before")

        y = nx.convert_matrix.to_scipy_sparse_matrix(self.graph, dtype=dtype, weight=None, format='csc')

        subject_vs = self.ndata[self.ndata.type == SUBJECT_TYPE].index.values
        paper_vs = self.ndata[self.ndata.type == PAPER_TYPE].index.values

        # Only consider subject subgraph
        y = y[:, subject_vs].tocsr()
        y = y[paper_vs, :]
        return y  # num of papers x num of subjects


#bgraph = BiblioGraph.from_tables(*load_tables(graph_dir, with_authors=False))

    
    
def load_data(graphdir, target_start_year, target_end_year, undirected=True, supervised=False, with_authors=True, collate_coauthorship=True,
              low_memory=False):

    bgraph = BiblioGraph.from_tables(*load_tables(graphdir, target_start_year=target_start_year, target_end_year=target_end_year, with_authors=with_authors, low_memory=low_memory),
                                     supervised=supervised,
                                     undirected=undirected,
                                     collate_coauthorship=collate_coauthorship)
    print(bgraph)
    print("Numericalize!")
    bgraph.numericalize_()

    print(bgraph)
    return bgraph
    



import torch
from torch_geometric.data import Dataset, Data, Batch
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import random

from config.data_config import DEFAULTS

class GraphDataset(Dataset):
    def __init__(self, data, task, data_config, use_for="training", transform=None, pre_transform=None):
        super().__init__(data, task, data_config, use_for, transform, pre_transform)

        ## - Get Task & Use For - ##
        self.task = task         
        self.use_for = use_for   

        ## - Process Data Config - ##
        self.data_config = {**DEFAULTS, **data_config}
        self.process_data_config(self.data_config)

        ## - Process Data - ##
        self.data = data
        self.process_data(self.data)
        
        ## - Process Graphs - ##
        self.graph_list = self.process_graphs()


    ## - Processing - ##

    def process_data_config(self, data_config):
        self.edges_to_use = data_config.get('edges_to_use')
        self.use_oneHot_parts = data_config.get('use_oneHot_parts')
        self.use_oneHot_types = data_config.get('use_oneHot_types')
        self.use_part_data = data_config.get('use_part_data')
        self.norm_part_data = data_config.get('norm_part_data')
        self.use_gta = data_config.get('use_gta')
        self.use_all_part_analytics = data_config.get('use_all_part_analytics')
        self.part_analytics_method = data_config.get('part_analytics_method')
        self.norm_gta = data_config.get('norm_gta')

    def process_data(self, data):
        ## - Design Data - ##
        self.design_data = data.get('training_data')
        self.scores_to_ranking = self.convert_scores_to_ranking() if self.task == 'ranking' else None
        
        ## - Interaction Library - ##
        self.interaction_library = data.get('interaction_library')

        ## - Part Library - ##
        self.part_library = pd.DataFrame(data.get('part_library'))
        self.process_part_library(self.part_library) if self.part_library is not None else None

    def process_part_library(self, part_library):
        self.mean_gta, self.std_gta = self.normalize_part_analytics() if (part_library and self.norm_gta) else (0, 1)
        self.parts = list(part_library.columns)
        self.part_types = part_library.loc['type'].tolist()

    
    ## - Build Graphs - ##

    def process_graphs(self):
        graph_data = [self.build_graph(row) for row in self.design_data]
        return graph_data

    def build_graph(self, row):
        edge_index, edge_attr = self.build_edges(row.get('design'))
        x = torch.tensor(self.build_nodes(row.get('design')))
        y = torch.tensor([self.get_label(row.get('score'))])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    
    
    ## - Build Edges - ##

    def build_edges(self, design):
        src, dest, edge_attr = self.build_interaction_edges(design) if self.edges_to_use in ['all', 'interaction'] else ([], [], [])
        src, dest, edge_attr += self.build_structure_edges(design, len(edge_attr[0])) if self.edges_to_use in ['all', 'structure'] else ([], [], [])
        src, dest, edge_attr += self.build_self_loops(design, len(edge_attr[0]))
        edge_index = [src, dest]
        return edge_index, edge_attr
    
    def build_interaction_edges(self, design):
        edge_attr = []
        src = []
        dest = []
        for index1, part1 in enumerate(design):
            if self.interaction_library.get(part1):
                for index2, part2 in enumerate(design):
                    interaction = self.interaction_library.get(part1).get(part2)
                    if interaction and (not interaction.get('only_adjacent') or index1+1 == index2):
                        src.append(index1)
                        dest.append(index2)
                        edge_attr.append(interaction.get('features'))
        return src, dest, edge_attr

    def build_structure_edges(self, design, edge_attr_size):
        src = [index for index in range(len(design) - 1)]
        dest = [index for index in range(1, len(design))]
        edge_attr = [list(np.zeros(edge_attr_size)) for i in range(len(design))]
        return src, dest, edge_attr

    def build_self_loops(self, design, edge_attr_size):
        src = [index for index in range(len(design))]
        dest = [index for index in range(len(design))]
        edge_attr = [list(np.zeros(edge_attr_size)) for i in range(len(design))]
        return src, dest, edge_attr

    
    ## - Build Nodes - ##

    def build_nodes(self, design):
        return [self.build_node_features(part, design[index+1] if index+1 < len(design) else None) for index, part in enumerate(design)]

    def build_node_features(self, part, next_part):
        node_features = []
        node_features += self.build_oneHot_parts(part) if self.use_oneHot_parts else []
        node_features += self.build_oneHot_types(part) if self.use_oneHot_types else []
        node_features += self.build_additional_node_features(part) if self.use_part_data else []
        node_features += self.build_gta_features(part, next_part) if self.use_gta else []
        return node_features if node_features else [1]

    def build_oneHot_parts(self, part):
        return [1 if part_id == part else 0 for part_id in self.parts]

    def build_oneHot_types(self, part):
        return [1 if (part_type == self.part_library.loc['type', part] and part_type) else 0 for part_type in self.part_types]

    def build_additional_node_features(self, part):
        return self.data.get('part_data').get(part)

    def build_gta_features(self, part, next_part):
        gta_features = [self.get_gta_feature(part, next_part)]
        
        if self.use_all_part_analytics:
            gta_features += [self.normalize_gta(float(self.part_library[part]['averageScore']))]
            gta_features += [self.normalize_gta(float(self.part_library[part]['lowScore']))]
            gta_features += [self.normalize_gta(float(self.part_library[part]['highScore']))]

        return gta_features
    
    def get_gta_feature(self, part, next_part):
        weights = self.part_library[part][next_part]
        if not weights or not next_part:
            return self.normalize_gta(float(self.part_library[part]['averageScore']))
        
        elif self.part_analytics_method == 'avg':
            return self.normalize_gta(np.mean(weights))
        
        elif self.part_analytics_method == 'random':
            return self.normalize_gta(random.choice(weights))
        
        else:
            return self.normalize_gta(float(self.part_library[part]['averageScore']))
    
    def normalize_gta(self, val):
        return (val - self.mean_gta) / self.std_gta
    
    def normalize_part_analytics(self):
        all_weights = [[float(weight) for weight in weights] for weights in self.part_library.loc['weights']]
        return (np.mean(all_weights), np.std(all_weights)) if len(all_weights) > 0 else (0, 1)
    

    ## - Labels - ##
    def get_label(self, score):
        if self.task == 'binary_classification':
            return 0 if score <= self.cut_off else 1
        if self.task == 'regression':
            return score
        if self.task == 'ranking':
            return self.scores_to_ranking.get(score, 0)
        
    def convert_scores_to_ranking(self):
        scores = []
        for row in self.design_data:
            scores.append(float(row['score']))

        ranks = rankdata([-s for s in scores], method='min')
        return dict(zip(scores, ranks))


    ## - Extras - ##

    def get_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.graph_list)

        return Batch.from_data_list(self.graph_list)
        
    def len(self):
        return len(self.graph_list)
    
    def get(self, index):
        return self.graph_list[index]
        
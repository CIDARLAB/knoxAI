import torch
from torch_geometric.data import Dataset, Data, Batch
import pandas as pd
import numpy as np
import random

from config.data_config import DEFAULTS

class GraphDataset(Dataset):
    def __init__(self, data, task, data_config, part_analytics=None, transform=None, pre_transform=None):
        super().__init__(data, task, data_config, part_analytics, transform, pre_transform)
        self.data = data
        self.task = task
        self.data_config = {**DEFAULTS, **data_config}
        self.part_analytics = pd.DataFrame(part_analytics) if part_analytics else None
        
        self.cut_off = np.median(data.get('scores')) if task == 'binary_classification' else None
        self.part_analytics = pd.DataFrame(part_analytics) if part_analytics else None
        self.mean_gta, self.std_gta = self.normalize_part_analytics() if part_analytics else (0, 1)
        self.score_to_ranking = self.data.get('score_to_ranking') if task == 'ranking' else None

        self.graph_list = self.process_data()
        
    def process_data(self):
        training_data = self.data.get('training_data')
        graph_data = [self.build_graph(row) for row in training_data]
        return graph_data

    def build_graph(self, row):
        edge_index, edge_attr = self.build_edges(row.get('design'))
        x = torch.tensor(self.build_nodes(row.get('design')))
        y = torch.tensor([self.get_label(row.get('score'))])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    
    def build_edges(self, design):
        edge_index, edge_attr = self.build_structure_edges(design) if self.data_config.get('edges_to_use') in ['all', 'structure'] else ([], [])
        edge_index, edge_attr += self.build_structure_edges(design) if self.data_config.get('edges_to_use') in ['all', 'structure'] else ([], [])
        return edge_index, edge_attr

    def build_structure_edges(self, design):
        edge_index = [[index for index in range(len(design) - 1)], [index for index in range(1, len(design))]] # [source, destination]
        edge_attr = [[0 for j in range(self.data.get('interaction_feature_size'))] for i in len(edge_index[0])]
        return edge_index, edge_attr

    def build_interaction_edges(self, design):
        edge_index = []
        edge_attr = []
        src = []
        dest = []
        for index1, part1 in enumerate(design):
            for index2, part2 in enumerate(design):
                interaction = self.data.get('interaction_library').get(part1).get(part2)
                if interaction and (not interaction.get('only_adjacent') or index1+1 == index2):
                    src.append(index1)
                    dest.append(index2)
                    edge_attr.append(interaction.get('features'))

        edge_index.append(src)
        edge_index.append(dest)
        return edge_index, edge_attr

    def build_nodes(self, design):
        return [self.build_node_features(part, design[index+1] if index+1 < len(design) else None) for index, part in enumerate(design)]

    def build_node_features(self, part, next_part):
        node_features = []
        node_features += self.build_oneHot_parts(part) if self.data_config.get('use_oneHot_parts') else []
        node_features += self.build_oneHot_types(part) if self.data_config.get('use_oneHot_types') else []
        node_features += self.build_additional_node_features(part) if self.data_config.get('use_part_data') else []
        node_features += self.build_gta_features(part, next_part) if self.data_config.get('use_gta') else []
        return node_features if node_features else [1]

    def build_oneHot_parts(self, part):
        return [1 if part_id == part else 0 for part_id in self.data.get('parts')]

    def build_oneHot_types(self, part):
        return [1 if part_type == self.data['part_library'][part] else 0 for part_type in self.data.get('part_types')]
    
    def build_additional_node_features(self, part):
        return self.data.get('part_data').get(part)

    def build_gta_features(self, part, next_part):
        gta_features = [self.get_gta_feature(part, next_part)]
        
        if self.data_config.get('use_all_part_analytics'):
            gta_features += [self.normalize_gta(float(self.part_analytics[part]['averageScore']))]
            gta_features += [self.normalize_gta(float(self.part_analytics[part]['lowScore']))]
            gta_features += [self.normalize_gta(float(self.part_analytics[part]['highScore']))]

        return gta_features
    
    def get_gta_feature(self, part, next_part):
        weights = self.part_analytics[part][next_part]
        if not weights or not next_part:
            return self.normalize_gta(float(self.part_analytics[part]['averageScore']))
        
        elif self.data_config.get('part_analytics_method') == 'avg':
            return self.normalize_gta(np.mean(weights))
        
        elif self.data_config.get('part_analytics_method') == 'random':
            return self.normalize_gta(random.choice(weights))
        
        else:
            return self.normalize_gta(float(self.part_analytics[part]['averageScore']))
    
    def normalize_gta(self, val):
        return (val - self.mean_gta) / self.std_gta
    
    def get_label(self, score):
        if self.task == 'binary_classification':
            return 0 if score <= self.cut_off else 1
        if self.task == 'regression':
            return score
        if self.task == 'ranking':
            return self.score_to_ranking.get(score, 0)

    def normalize_part_analytics(self):
        all_weights = [[float(weight) for weight in weights] for weights in self.part_analytics.loc['weights']]
        return (np.mean(all_weights), np.std(all_weights)) if len(all_weights) > 0 else (0, 1)
    
    def get_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.graph_list)

        return Batch.from_data_list(self.graph_list)
        
    def len(self):
        return len(self.graph_list)
    
    def get(self, index):
        return self.graph_list[index]
        
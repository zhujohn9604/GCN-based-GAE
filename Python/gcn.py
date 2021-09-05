"""
File: gcn_cv_sc.py
Author: Lukas
Email: vim@lpag.de
Github: https://github.com/lgalke
Description: GCNs with Control Variate and Skip-Connections
Derived from: https://raw.githubusercontent.com/dmlc/dgl/master/examples/mxnet/sampling/gcn_cv_sc.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse, time, math
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class NodeUpdate(nn.Module):
    def __init__(self,
                 layer_id,
                 in_feats,
                 out_feats,
                 dropout,
                 activation=None,
                 concat=False):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.concat = concat
        self.dense = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = node.data['h']
        if not self.training:
            # Instead of averaging, mult with norm
            norm = node.data['norm'].to(h.device)
            h = h * norm
        else:
            agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
            agg_history = node.data[agg_history_str].to(h.device)
            # control variate
            h = h + agg_history
            if self.dropout:
                h = self.dropout(h)
        h = self.dense(h)
        if self.concat:
            h = torch.cat([h, self.activation(h)], dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}



class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.activation = nn.ReLU()
        #self.activation = None
        #self.activation = activation
        self.n_layers = int(n_layers)

        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(NodeUpdate(1,
                                          in_feats,
                                          n_classes,
                                          dropout,
                                          activation,
                                          concat=False))
                                          # Don't use skip connections if we have only 1 layer
        else:
            # input layer
            self.layers.append(NodeUpdate(1,
                                          in_feats,
                                          n_hidden,
                                          dropout,
                                          activation,
                                          concat=(n_layers == 2)))
            for i in range(2, n_layers):
                skip_start = (i == self.n_layers-1)
                self.layers.append(NodeUpdate(i,
                                              n_hidden,
                                              n_hidden,
                                              dropout,
                                              activation,
                                              concat=skip_start))

            self.layers.append(NodeUpdate(n_layers,
                                          2*n_hidden,
                                          n_classes,
                                          dropout))

    def forward(self, nf):
        # get embedding and put it on same device as input layer
        my_device = self.layers[0].dense.weight.device

        h = nf.layers[0].data['embed']

        out_device = h.device

        h = h.to(my_device)


        for i, layer in enumerate(self.layers):
            # Debug: 850k samples drawn for 'Theory'
            # print("NF layer_size[%d]" % i, nf.layer_size(i))
            if self.training:
                new_history = h.clone().detach()
                history_str = 'h_{}'.format(i)
                history = nf.layers[i].data[history_str].to(my_device)
                h = h - history

                # .data['h'] is now on my_device
                nf.layers[i].data['h'] = h
                nf.block_compute(i,
                                 fn.copy_src(src='h', out='m'),
                                 lambda node: {'h': node.mailbox['m'].mean(dim=1)},
                                 layer)
                h = nf.layers[i+1].data.pop('activation')
                # update history
                if i < nf.num_layers-1:
                    nf.layers[i].data[history_str] = new_history.to(out_device)
            else:
                # .data['h'] is now on my_device
                nf.layers[i].data['h'] = h
                nf.block_compute(i,
                                 fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'),  # Is it right to use sum here?
                                 layer)                     # Orig impl does it...
                h = nf.layers[i+1].data.pop('activation')
        return h